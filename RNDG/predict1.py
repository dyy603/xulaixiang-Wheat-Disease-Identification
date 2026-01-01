import os
import json
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from model3 import create_regnet_with_attention
import datetime
import argparse


def load_custom_font(font_size=18):
    """尝试加载自定义字体，失败则使用默认字体"""
    try:
        # 尝试加载中文字体，这里使用微软雅黑，你可以根据需要修改路径
        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
            "C:/Windows/Fonts/simhei.ttf",  # 黑体
            "C:/Windows/Fonts/simsun.ttc",  # 宋体
            "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",  # Linux字体
        ]

        for font_path in font_paths:
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, font_size)

        # 如果都没找到，尝试系统默认字体
        return ImageFont.truetype("arial.ttf", font_size)
    except:
        # 如果都失败，使用PIL默认字体
        return ImageFont.load_default()


def add_prediction_to_image(img, prediction_result, class_indict):
    """将预测结果添加到图片上"""
    # 创建图片副本，避免修改原图
    img_with_text = img.copy()
    draw = ImageDraw.Draw(img_with_text)

    # 获取图片尺寸
    img_width, img_height = img.size

    # 根据图片大小动态调整字体大小
    base_font_size = min(img_width, img_height) // 30
    font_size = max(16, min(base_font_size, 24))  # 限制在16-24之间

    # 加载字体
    font = load_custom_font(font_size)

    # 设置颜色
    text_color = (255, 255, 255)  # 白色
    bg_color = (0, 0, 0, 180)  # 半透明黑色背景

    # 预测结果文本 - 简化格式
    result_text = f"class: {prediction_result['predicted_class']}, acc: {prediction_result['confidence']:.2%}"

    # 计算文本区域大小
    text_bbox = draw.textbbox((0, 0), result_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # 计算信息框尺寸
    padding = 10
    info_width = text_width + padding * 2
    info_height = text_height + padding * 2

    # 确保信息框不会超过图片宽度
    if info_width > img_width:
        info_width = img_width
        # 缩小字体再试一次
        font = load_custom_font(font_size - 2)
        text_bbox = draw.textbbox((0, 0), result_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        info_width = text_width + padding * 2
        info_height = text_height + padding * 2

    # 创建半透明背景
    bg_img = Image.new('RGBA', (info_width, info_height), bg_color)

    # 将背景放在图片左上角
    position = (10, 10)
    img_with_text.paste(bg_img, position, bg_img)

    # 绘制文本
    draw.text((position[0] + padding, position[1] + padding),
              result_text, font=font, fill=text_color)

    # 如果图片足够大，在底部显示所有类别的概率
    if img_height > 300:
        # 计算底部信息框位置
        bottom_height = 30 * (len(class_indict) + 1)  # 每行30像素，加标题行
        bottom_bg = Image.new('RGBA', (img_width, bottom_height), bg_color)
        img_with_text.paste(bottom_bg, (0, img_height - bottom_height), bottom_bg)

        # 使用更小的字体显示概率
        small_font_size = max(12, min(base_font_size - 4, 18))
        small_font = load_custom_font(small_font_size)

        # 添加标题
        draw.text((10, img_height - bottom_height + 5),
                  "各类别概率:", font=small_font, fill=text_color)

        # 显示每个类别的概率
        sorted_probs = sorted(prediction_result['all_probabilities'].items(),
                              key=lambda x: x[1], reverse=True)

        for i, (class_name, prob) in enumerate(sorted_probs):
            y_pos = img_height - bottom_height + 30 + i * 25
            prob_text = f"{class_name}: {prob:.2%}"
            draw.text((10, y_pos), prob_text, font=small_font, fill=text_color)

    return img_with_text


def predict_single_image(model, image_path, transform, class_indict, device):
    """预测单张图片"""
    try:
        # 打开并预处理图片
        img = Image.open(image_path).convert('RGB')
        original_img = img.copy()  # 保存原始图片用于显示

        img_tensor = transform(img).unsqueeze(0).to(device)

        # 预测
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            _, pred = torch.max(output, 1)
            pred_label = pred.item()

        # 获取预测类别名称和置信度
        class_name = class_indict.get(str(pred_label), f"未知类别{pred_label}")
        confidence = probabilities[0][pred_label].item()

        # 获取所有类别的概率
        all_probabilities = {}
        for i in range(len(class_indict)):
            class_name_i = class_indict.get(str(i), f"未知类别{i}")
            all_probabilities[class_name_i] = probabilities[0][i].item()

        return {
            'predicted_class': class_name,
            'predicted_label': pred_label,
            'confidence': confidence,
            'all_probabilities': all_probabilities,
            'image_path': image_path,
            'original_image': original_img,
            'success': True
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'image_path': image_path
        }


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='使用RegNet模型进行单张图片预测')
    parser.add_argument('--image', type=str,
                        default=r'D:\regnet_xiaomai\wheat3\test\Healthy\zc507.jpg',
                        help='要预测的图片路径')
    parser.add_argument('--save_image', action='store_true', default=True,
                        help='保存带预测结果的图片')
    parser.add_argument('--output_dir', type=str,
                        default=r'D:\regnet_xiaomai\results-best\predicted_images',
                        help='保存结果图片的目录')
    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_type = "GPU" if torch.cuda.is_available() else "CPU"
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"单张图片预测 on {device_type} at {current_time}")
    print(f"图片路径: {args.image}")
    print("-" * 50)

    # 数据预处理（与训练时验证集保持一致）
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载类别索引（确保包含4类）
    json_path = r'D:\regnet_xiaomai\class_indices.json'  # 确保该文件包含4类
    assert os.path.exists(json_path), f"类别索引文件 {json_path} 不存在"
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 验证类别数是否为4
    assert len(class_indict) == 4, f"类别索引文件中包含 {len(class_indict)} 类，预期为4类"
    num_classes = 4  # 显式指定为4类

    # 创建模型（指定4类）
    model = create_regnet_with_attention(
        model_name="regnety_400mf",
        num_classes=num_classes  # 确保输出为4类
    ).to(device)

    # 模型权重路径
    model_weight_path = r"D:\regnet_xiaomai\weights3\best_model.pth"
    assert os.path.exists(model_weight_path), f"权重文件 {model_weight_path} 不存在"

    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    print(f"模型加载自: {model_weight_path}")
    print("-" * 50)

    # 检查图片是否存在
    if not os.path.exists(args.image):
        print(f"错误: 图片路径不存在: {args.image}")
        print("请使用 --image 参数指定正确的图片路径")
        return

    # 执行预测
    result = predict_single_image(model, args.image, data_transform, class_indict, device)

    if result['success']:
        # 控制台输出
        print(f"预测结果:")
        print(f"  class: {result['predicted_class']}")
        print(f"  acc: {result['confidence']:.2%}")

        print("\n所有类别概率:")
        for class_name, prob in result['all_probabilities'].items():
            print(f"  {class_name}: {prob:.2%}")

        # 将预测结果添加到图片上
        if args.save_image:
            annotated_image = add_prediction_to_image(
                result['original_image'],
                result,
                class_indict
            )

            # 确保输出目录存在
            os.makedirs(args.output_dir, exist_ok=True)

            # 生成输出文件名
            original_filename = os.path.basename(result['image_path'])
            filename_without_ext = os.path.splitext(original_filename)[0]
            output_filename = f"{filename_without_ext}_predicted.jpg"
            output_path = os.path.join(args.output_dir, output_filename)

            # 保存图片
            annotated_image.save(output_path)
            print(f"\n带预测结果的图片已保存到: {output_path}")

            # 显示图片
            try:
                annotated_image.show()
            except:
                print("注意: 无法自动显示图片，请查看保存的文件")
    else:
        print(f"预测失败: {result['error']}")

    print("-" * 50)

    # 保存文本结果
    results_dir = r"D:\regnet_xiaomai\results-best"
    os.makedirs(results_dir, exist_ok=True)

    # 结果文件路径
    single_prediction_file = os.path.join(results_dir, "single_prediction_results.txt")

    # 保存结果到文件
    with open(single_prediction_file, 'a', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"预测时间: {current_time}\n")
        f.write(f"设备: {device_type}\n")
        f.write(f"模型权重: {os.path.basename(model_weight_path)}\n")
        f.write(f"图片路径: {result['image_path']}\n")

        if result['success']:
            f.write(f"class: {result['predicted_class']}\n")
            f.write(f"acc: {result['confidence']:.2%}\n")

            if args.save_image:
                f.write(f"带预测结果的图片: {output_path}\n")

            f.write("所有类别概率:\n")
            for class_name, prob in result['all_probabilities'].items():
                f.write(f"  {class_name}: {prob:.2%}\n")
        else:
            f.write(f"预测失败: {result['error']}\n")

        f.write("=" * 60 + "\n\n")

    print(f"文本结果已追加到: {single_prediction_file}")


if __name__ == '__main__':
    main()