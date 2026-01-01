import os
import math
import argparse
import torch
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from model3 import create_regnet_with_attention
from my_dataset import MyDataSet
from utils import train_one_epoch, evaluate


def get_images_and_labels(data_path):
    """从指定路径读取图像路径和对应的标签"""
    classes = os.listdir(data_path)
    images_path = []
    images_label = []
    for i, cls in enumerate(classes):
        cls_path = os.path.join(data_path, cls)
        images = [os.path.join(cls_path, img) for img in os.listdir(cls_path)]
        images_path.extend(images)
        images_label.extend([i] * len(images))
    return images_path, images_label


def calculate_metrics(model, data_loader, device, criterion):
    """计算指定数据集的损失和准确率"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    weights_dir = "./weights3"
    if os.path.exists(weights_dir):
        for f in os.listdir(weights_dir):
            if f.endswith(".pth"):
                os.remove(os.path.join(weights_dir, f))
    else:
        os.makedirs(weights_dir, exist_ok=True)

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    if not os.path.exists("./weights3"):
        os.makedirs("./weights3")

    # 创建日志文件，包含train/val/test的loss和acc
    log_file_path = "./training3.txt"
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        log_file.write("epoch\t"
                       "train_loss\t train_acc\t"
                       "val_loss\t val_acc\t"
                       "test_loss\t test_acc\t"
                       "learning_rate\n")

        # 加载训练集、验证集、测试集
        train_images_path, train_images_label = get_images_and_labels(os.path.join(args.data_path, 'train'))
        val_images_path, val_images_label = get_images_and_labels(os.path.join(args.data_path, 'val'))
        test_images_path, test_images_label = get_images_and_labels(os.path.join(args.data_path, 'test'))  # 新增测试集

        data_transform = {
            "train": transforms.Compose([
                transforms.RandomVerticalFlip(p=0.2),
                transforms.Resize((224, 224)),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.5)
            ]),
            "val": transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            "test": transforms.Compose([  # 测试集使用与验证集相同的预处理
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

        # 创建数据集和数据加载器
        train_dataset = MyDataSet(train_images_path, train_images_label, data_transform["train"])
        val_dataset = MyDataSet(val_images_path, val_images_label, data_transform["val"])
        test_dataset = MyDataSet(test_images_path, test_images_label, data_transform["test"])  # 新增测试集

        batch_size = args.batch_size
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
        print(f'Using {nw} dataloader workers every process')

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=nw,
                                                   collate_fn=train_dataset.collate_fn)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=nw,
                                                 collate_fn=val_dataset.collate_fn)
        test_loader = torch.utils.data.DataLoader(test_dataset,  # 新增测试集加载器
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=nw,
                                                  collate_fn=test_dataset.collate_fn)

        # 模型初始化
        model = create_regnet_with_attention(model_name=args.model_name,
                                             num_classes=args.num_classes).to(device)
        if args.weights != "":
            if os.path.exists(args.weights):
                weights_dict = torch.load(args.weights, map_location=device)
                adjusted_weights_dict = {}
                for k, v in weights_dict.items():
                    if k.startswith('head.'):
                        adjusted_k = 'head.1.' + k[len('head.'):]
                        if adjusted_k in model.state_dict():
                            adjusted_weights_dict[adjusted_k] = v
                    else:
                        if k in model.state_dict():
                            adjusted_weights_dict[k] = v
                load_weights_dict = {k: v for k, v in adjusted_weights_dict.items()
                                     if model.state_dict()[k].numel() == v.numel()}
                model.load_state_dict(load_weights_dict, strict=False)
            else:
                raise FileNotFoundError(f"未找到权重文件: {args.weights}")

        if args.freeze_layers:
            for name, para in model.named_parameters():
                if "head" not in name:
                    para.requires_grad_(False)
                else:
                    print(f"train {name}")

        # 优化器和损失函数
        pg = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)

        best_val_acc = 0

        for epoch in range(args.epochs):
            # 训练阶段
            train_loss = train_one_epoch(model=model,
                                         optimizer=optimizer,
                                         data_loader=train_loader,
                                         device=device,
                                         epoch=epoch)

            # 计算各数据集指标
            train_acc = evaluate(model, train_loader, device)  # 训练集准确率
            val_loss, val_acc = calculate_metrics(model, val_loader, device, criterion)  # 验证集loss和acc
            test_loss, test_acc = calculate_metrics(model, test_loader, device, criterion)  # 测试集loss和acc

            # 学习率调度
            scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]["lr"]

            # 打印指标
            print(f"[epoch {epoch}] "
                  f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
                  f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, "
                  f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}, "
                  f"lr: {current_lr:.6f}")

            # TensorBoard记录
            tb_writer.add_scalar("train_loss", train_loss, epoch)
            tb_writer.add_scalar("train_accuracy", train_acc, epoch)
            tb_writer.add_scalar("val_loss", val_loss, epoch)
            tb_writer.add_scalar("val_accuracy", val_acc, epoch)
            tb_writer.add_scalar("test_loss", test_loss, epoch)
            tb_writer.add_scalar("test_accuracy", test_acc, epoch)
            tb_writer.add_scalar("learning_rate", current_lr, epoch)

            # 保存模型
            torch.save(model.state_dict(), f"./weights3/model-{epoch}.pth")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "./weights3/best_model.pth")
                print(f"保存最佳模型，验证准确率: {best_val_acc:.4f}")

            # 写入日志文件
            log_file.write(f"{epoch}\t"
                           f"{train_loss:.4f}\t{train_acc:.4f}\t"
                           f"{val_loss:.4f}\t{val_acc:.4f}\t"
                           f"{test_loss:.4f}\t{test_acc:.4f}\t"
                           f"{current_lr:.6f}\n")
            log_file.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--data-path', type=str,
                        default=r"D:\regnet_xiaomai\wheat3")  # 需确保该路径下有test文件夹
    parser.add_argument('--model-name', default='regnety_400mf')
    parser.add_argument('--weights', type=str, default=r'D:\regnet_xiaomai\regnety_400mf.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    main(opt)