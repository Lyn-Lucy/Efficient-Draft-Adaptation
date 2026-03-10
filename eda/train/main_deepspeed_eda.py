import argparse
import deepspeed
import shutil

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--basepath', type=str, default='/home/lyh/weights/hf/llama3chat/8B/')
parser.add_argument('--tmpdir', type=str,
                    default='/home/lyh/code/nlp/ess/feature_data_dataset/sharegpt_0_67999_mu_V7B/')
parser.add_argument('--cpdir', type=str, default='0')
parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
# EDA 新增参数
parser.add_argument('--transfer', action='store_true', help='Enable transfer training mode')
parser.add_argument('--pretrained_eda', type=str, default=None, help='Path to pretrained EDA checkpoint for transfer')
parser.add_argument('--freeze_attention', action='store_true', help='Also freeze attention layers during transfer')
parser.add_argument('--freeze_fc', action='store_true', help='Also freeze fc layer during transfer')
parser.add_argument('--private_intermediate_size', type=int, default=None, help='Intermediate size for private experts (default: same as config.intermediate_size)')
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()
import json

train_config = {
    "lr": 5e-5,
    "bs": 4,
    "gradient_accumulation_steps": 1,
    "datapath": f"{args.tmpdir}",
    "is_warmup": True,
    "num_epochs": 20,
    "num_warmup_steps": 2000,
    "total_steps": 800000,
    "p_w": 0.1,
    "v_w": 1.0,
    "head_w": 0.1,
    # EDA 新增配置
    "shared_aux_loss_w": 0,
    "private_aux_loss_w": 0,
    "num_shared_experts": 1,
    "num_private_experts": 1,
    "top_k_shared": 1,
    "top_k_private": 1,
    "num_workers": 2,
    "embeding": True,
    "act": "No",
    "data_noise": True,
    "noise": "uniform",
    "mean": 0.0,
    "std": 0.2,
    "residual": "true,norm",
    "max_len": 2048,
    "config_path": "qwen2_7B_config.json",
    "b1": 0.9,
    "b2": 0.95,
    "grad_clip": 0.5,
}

from safetensors import safe_open
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch

torch.backends.cuda.matmul.allow_tf32 = True
from accelerate import Accelerator
from accelerate.utils import set_seed

set_seed(0)
accelerator = Accelerator(mixed_precision="bf16")
from eda.model.cnets_eda import Model, load_balancing_loss_func
from eda.model.configs import EConfig
from typing import Any, Dict, List

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
# import accelerate
import numpy as np


deepspeed.init_distributed()
rank = torch.distributed.get_rank()

try:
    with open(os.path.join(args.basepath, "model.safetensors.index.json"), "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["lm_head.weight"]
    with safe_open(os.path.join(args.basepath, head_path),
                   framework="pt",
                   device="cpu") as f:
        tensor_slice = f.get_slice("lm_head.weight")
        vocab_size, hidden_dim = tensor_slice.get_shape()
        tensor = tensor_slice[:, :hidden_dim].bfloat16()
except:
    with open(os.path.join(args.basepath, "pytorch_model.bin.index.json"), "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["lm_head.weight"]
    weights = torch.load(os.path.join(args.basepath, head_path))
    tensor = weights["lm_head.weight"].bfloat16()

head = torch.nn.Linear(tensor.shape[1], tensor.shape[0], bias=False, dtype=torch.bfloat16)
head.weight.data = tensor


def list_files(path):
    datapath = []
    for root, directories, files in os.walk(path, followlinks=True):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.0):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class AddUniformNoise:
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class CustomDataset(Dataset):
    def __init__(self, datapath, transform=None):
        self.data = datapath
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Try to load data, skip corrupted files
        try:
            data = torch.load(self.data[index])
        except Exception as e:
            print(f"Warning: Failed to load {self.data[index]}: {e}")
            # Return a dummy sample or try next index
            return self.__getitem__((index + 1) % len(self.data))
        
        new_data = {}
        hidden_state = data['hidden_state'][:train_config["max_len"]][None, :]
        input_ids = data['input_ids'][:train_config["max_len"]][None, :]
        loss_mask = data["loss_mask"][:train_config["max_len"]][None, :]


        length = hidden_state.shape[1]
        # length_q = data['query_ids'].shape[1]
        attention_mask = [1] * length
        loss_mask = loss_mask[0].tolist()
        loss_mask[-1] = 0

        input_ids_target = input_ids[:, 1:]
        zeropadding = torch.tensor([[0]])
        input_ids_target = torch.cat((input_ids_target, zeropadding), dim=1)

        target = hidden_state[:, 1:, :]
        zeropadding = torch.zeros(1, 1, target.shape[2])
        target = torch.cat((target, zeropadding), dim=1)
        loss_mask[-1] = 0
        new_data["attention_mask"] = attention_mask
        new_data["loss_mask"] = loss_mask
        new_data["target"] = target
        new_data["hidden_state_big"] = hidden_state
        new_data["input_ids"] = input_ids_target


        if self.transform:
            new_data = self.transform(new_data)

        return new_data


class DataCollatorWithPadding:

    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        # padding_tensor = torch.zeros(B, N - n, S,dtype=intensors.dtype)
        padding_tensor = torch.zeros(B, N - n, S)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['hidden_state_big'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_hidden_states = torch.cat([self.paddingtensor(item['hidden_state_big'], max_length) for item in features])
        batch_target = torch.cat([self.paddingtensor(item['target'], max_length) for item in features])
        batch_loss_mask = torch.tensor(
            [item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features])
        batch_attention_mask = torch.tensor(
            [item['attention_mask'] + [0] * (max_length - len(item['attention_mask'])) for item in features])
        # batch_loss_mask = torch.ones_like(batch_loss_mask)
        # batch_attention_mask=torch.ones_like(batch_attention_mask)
        batch = {
            "input_ids": batch_input_ids,
            "hidden_states": batch_hidden_states,
            "target": batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch


def top_accuracy(output, target, topk=(1,)):
    # output.shape (bs, num_classes), target.shape (bs, )
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res

def compute_loss(target, target_p, predict, loss_mask, shared_router_logits=None, private_router_logits=None):
    out_head = head_engine(predict)
    out_logp = nn.LogSoftmax(dim=2)(out_head)
    plogp = target_p * out_logp
    ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / (loss_mask.shape[0] * loss_mask.shape[1])
    # Use float32 for SmoothL1Loss as it doesn't support bfloat16 backward
    vloss = criterion(predict.float(), target.to(rank).float())
    vloss = torch.sum(torch.mean(loss_mask.float() * vloss, 2)) / (loss_mask.shape[0] * loss_mask.shape[1])
    
    # EDA: 计算两组专家的负载均衡 loss
    shared_aux_loss = torch.tensor(0.0, device=rank)
    private_aux_loss = torch.tensor(0.0, device=rank)
    
    if shared_router_logits is not None and len(shared_router_logits) > 0:
        shared_aux_loss = load_balancing_loss_func(
            shared_router_logits, 
            num_experts=train_config["num_shared_experts"], 
            top_k=train_config["top_k_shared"]
        )
    
    if private_router_logits is not None and len(private_router_logits) > 0:
        private_aux_loss = load_balancing_loss_func(
            private_router_logits, 
            num_experts=train_config["num_private_experts"], 
            top_k=train_config["top_k_private"]
        )
    
    return vloss, ploss, shared_aux_loss, private_aux_loss, out_head



if train_config["data_noise"]:
    if train_config["noise"] == "uniform":
        aug = AddUniformNoise(std=train_config["std"])
    else:
        aug = AddGaussianNoise(mean=train_config["mean"], std=train_config["std"])
else:
    aug = None

datapath = list_files(train_config["datapath"])

traindatapath = datapath[:int(len(datapath) * 0.95)]
testdatapath = datapath[int(len(datapath) * 0.95):]
traindataset = CustomDataset(traindatapath, transform=aug)
testdataset = CustomDataset(testdatapath)
test_loader = DataLoader(testdataset, batch_size=train_config["bs"], shuffle=False,
                         collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"], pin_memory=True)


if rank == 0:
    if not os.path.exists(args.cpdir):
        os.makedirs(args.cpdir)
    
    # 保存训练配置和代码文件以便记录
    print(f"\nSaving training configurations and code files to {args.cpdir}")
    
    # 保存训练参数
    config_save_path = os.path.join(args.cpdir, "train_args.json")
    args_dict = vars(args)
    args_dict['train_config'] = train_config
    with open(config_save_path, 'w', encoding='utf-8') as f:
        json.dump(args_dict, f, indent=2, ensure_ascii=False)
    print(f"  - Saved training arguments to: train_args.json")
    
    # 保存训练脚本
    current_file = os.path.abspath(__file__)
    train_script_save_path = os.path.join(args.cpdir, "main_deepspeed_eda.py")
    shutil.copy(current_file, train_script_save_path)
    print(f"  - Saved training script to: main_deepspeed_eda.py")
    
    # 保存模型文件
    model_file = os.path.join(os.path.dirname(os.path.dirname(current_file)), "model", "cnets_eda.py")
    if os.path.exists(model_file):
        model_save_path = os.path.join(args.cpdir, "cnets_eda.py")
        shutil.copy(model_file, model_save_path)
        print(f"  - Saved model file to: cnets_eda.py")
    
    print(f"Configuration and code files saved successfully!\n")

config = EConfig.from_pretrained(train_config["config_path"])
# EDA 配置
config.num_shared_experts = train_config["num_shared_experts"]
config.num_private_experts = train_config["num_private_experts"]
config.top_k_shared = train_config["top_k_shared"]
config.top_k_private = train_config["top_k_private"]
# 私有专家 intermediate_size（默认与 config.intermediate_size 相同）
if args.private_intermediate_size is not None:
    config.private_intermediate_size = args.private_intermediate_size
    if rank == 0:
        print(f"Private experts intermediate_size set to: {args.private_intermediate_size}")
else:
    config.private_intermediate_size = config.intermediate_size

model = Model(config, path=args.basepath, load_emb=True)

# 迁移训练模式
if args.transfer and args.pretrained_eda:
    if rank == 0:
        print(f"\n{'='*60}")
        print("Transfer Training Mode")
        print(f"{'='*60}")
        print(f"Loading pretrained shared experts from: {args.pretrained_eda}")
    
    model.load_shared_experts(args.pretrained_eda)
    model.freeze_shared_experts()
    
    if args.freeze_attention:
        model.freeze_attention()
    
    if args.freeze_fc:
        model.freeze_fc()
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nParameter Statistics:")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        print(f"  - Frozen parameters: {total_params - trainable_params:,}")
        print(f"  - Trainable ratio: {trainable_params / total_params * 100:.2f}%")
        print(f"{'='*60}\n")
else:
    if rank == 0:
        print(f"\n{'='*60}")
        print("Full Training Mode")
        print(f"{'='*60}")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"EDA Model: {total_params:,} parameters")
        print(f"  - {train_config['num_shared_experts']} shared experts (top-{train_config['top_k_shared']})")
        print(f"  - {train_config['num_private_experts']} private experts (top-{train_config['top_k_private']})")
        print(f"{'='*60}\n")

criterion = nn.SmoothL1Loss(reduction="none")

num_epochs = train_config["num_epochs"]
num_warmup_steps = train_config["num_warmup_steps"]
total_steps = train_config["total_steps"]
is_warmup = train_config["is_warmup"]

# 只传递可训练参数给优化器
trainable_params = [p for p in model.parameters() if p.requires_grad]

model_engine, optimizer, train_loader, _ = deepspeed.initialize(args=args,
                                                                model=model,
                                                                model_parameters=trainable_params,
                                                                training_data=traindataset,
                                                                collate_fn=DataCollatorWithPadding()
                                                                )

head_engine, _, test_loader, _ = deepspeed.initialize(args=args,
                                                      model=head,
                                                      model_parameters=head.parameters(),
                                                      training_data=testdataset,
                                                      collate_fn=DataCollatorWithPadding()
                                                      )


for param in head.parameters():
    param.requires_grad = False

for epoch in range(num_epochs):
    top_3acc = [0 for _ in range(3)]
    correct = 0
    total = 0
    epoch_loss = 0
    num_batches = 0
    model.train()
    
    # Use tqdm for progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=(rank != 0))
    
    for batch_idx, data in enumerate(pbar):

        model.zero_grad()

        # EDA: 输出 router logits
        predict, shared_router_logits, private_router_logits = model_engine(
            data["hidden_states"].to(rank).bfloat16(), 
            input_ids=data["input_ids"].to(rank),
            attention_mask=data["attention_mask"].to(rank),
            output_router_logits=True
        )
        with torch.no_grad():
            target_head = head_engine(data["target"].to(rank).bfloat16())
            target_p = nn.Softmax(dim=2)(target_head)
            target_p = target_p.detach()

        loss_mask = data["loss_mask"][:, :, None].to(rank)
        vloss, ploss, shared_aux_loss, private_aux_loss, out_head = compute_loss(
            data["target"], target_p, predict, loss_mask, shared_router_logits, private_router_logits
        )
        
        # EDA: 总 loss 包含两组专家的负载均衡 loss
        loss = (train_config["v_w"] * vloss + 
                train_config["p_w"] * ploss + 
                train_config["shared_aux_loss_w"] * shared_aux_loss +
                train_config["private_aux_loss_w"] * private_aux_loss)
        # loss.backward()
        model_engine.backward(loss)
        # accelerator.clip_grad_value_(model.parameters(), train_config["grad_clip"])

        model_engine.step()

        with torch.no_grad():
            _, predicted = torch.max(out_head, 2)
            _, target = torch.max(target_head, 2)
            ct = loss_mask.sum().item()
            cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
            out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
            target = target.view(-1)[loss_mask.view(-1) == 1]
            topkacc = top_accuracy(out_head, target, (1, 2, 3))
            for top_i in range(len(topkacc)):
                top_3acc[top_i] += topkacc[top_i]
            total += ct
            correct += cc
        if rank == 0 and ct != 0:
            logdict = {"train/lr": optimizer.optimizer.param_groups[0]["lr"], "train/vloss": vloss.item(),
                       "train/ploss": ploss.item(), "train/loss": loss.item(), "train/acc": cc / ct,
                       "train/shared_aux_loss": shared_aux_loss.item(), "train/private_aux_loss": private_aux_loss.item()}
            for id, i in enumerate(top_3acc):
                logdict[f'train/top_{id + 1}_acc'] = topkacc[id].item() / ct
            # Update progress bar with metrics
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'vloss': f'{vloss.item():.4f}',
                'ploss': f'{ploss.item():.4f}',
                'acc': f'{cc/ct:.4f}',
                's_aux': f'{shared_aux_loss.item():.4f}',
                'p_aux': f'{private_aux_loss.item():.4f}'
            })

        del ploss, vloss, shared_aux_loss, private_aux_loss
        epoch_loss += loss.item()
        num_batches += 1

    correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
    correct, total = accelerator.gather_for_metrics((correct, total))
    correct, total = correct.sum().item(), total.sum().item()
    epoch_loss /= num_batches
    top_3acc = accelerator.gather_for_metrics(top_3acc)
    if accelerator.is_local_main_process:
        for id, i in enumerate(top_3acc):
            print(f'Epoch {epoch+1} top_{id + 1}_acc: {i.sum().item() / total:.4f}')
    if accelerator.is_local_main_process:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
        print('Train Accuracy: {:.2f}%'.format(100 * correct / (total + 1e-5)))

    save_path = f"{args.cpdir}/state_{epoch}"
    model_engine.save_16bit_model(save_path)
    # 保存 config.json
    if rank == 0:
        # 保存 EDA 配置
        eda_config = {
            "architectures": ["SPMoEModel"],
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_attention_heads": config.num_attention_heads,
            "num_hidden_layers": config.num_hidden_layers,
            "num_key_value_heads": config.num_key_value_heads,
            "vocab_size": config.vocab_size,
            "rms_norm_eps": config.rms_norm_eps,
            "max_position_embeddings": config.max_position_embeddings,
            "hidden_act": config.hidden_act,
            "rope_theta": getattr(config, 'rope_theta', 10000.0),
            "num_shared_experts": train_config["num_shared_experts"],
            "num_private_experts": train_config["num_private_experts"],
            "top_k_shared": train_config["top_k_shared"],
            "top_k_private": train_config["top_k_private"],
            "training_mode": "transfer" if args.transfer else "full",
            "pretrained_eda": args.pretrained_eda if args.transfer else None,
        }
        config_dst = os.path.join(save_path, "config.json")
        with open(config_dst, 'w') as f:
            json.dump(eda_config, f, indent=2)
        print(f"Config saved to {config_dst}")
    if epoch % 10 == 0:
        deepspeed.DeepSpeedEngine.save_checkpoint(model_engine, save_dir=save_path)
