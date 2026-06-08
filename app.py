"""
XiaoLeiLuan
"""
import os
import io
import json
import logging
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

# 使用 PyTorch 官方 ResNet18
from torchvision.models import resnet18

# ==================== 配置 ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {DEVICE}")

# ==================== 加载类别索引 ====================
CLASS_INDICES_PATH = os.path.join(BASE_DIR, "class_indices.json")
CLASS_INDICES = {}
CLASS_NAMES = []  # 按索引顺序的英文类名

try:
    with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
        CLASS_INDICES = json.load(f)
    # 按键排序确保顺序
    sorted_items = sorted(CLASS_INDICES.items(), key=lambda x: int(x[0]))
    CLASS_NAMES = [item[1] for item in sorted_items]
    logger.info(f"已加载 {len(CLASS_NAMES)} 个 PlantVillage 标准类别")
except Exception as e:
    logger.error(f"class_indices.json 加载失败: {e}")

# ==================== 加载 ResNet 模型 ====================
MODEL_PATH = os.path.join(BASE_DIR, "best_model_process (1).pth")
IMG_SIZE = 224
model = None
MODEL_READY = False

# 推理时预处理
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

logger.info("正在加载 ResNet18 44类分类模型...")

try:
    model = resnet18(weights=None, num_classes=len(CLASS_NAMES))
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model = model.to(DEVICE)
    MODEL_READY = True
    logger.info(f"ResNet18 44类模型加载成功!")
except FileNotFoundError:
    logger.warning(f"模型文件未找到: {MODEL_PATH}")
    logger.warning("当前使用 HSV 图像分析模式")
except Exception as e:
    logger.warning(f"模型加载失败: {e}，当前使用 HSV 图像分析模式")

# ==================== 英文类名 → 中文类名 映射 ====================
EN_TO_CN = {
    "Apple_Apple_scab": "苹果黑星病",
    "Apple_Black_rot": "苹果黑腐病",
    "Apple_Cedar_apple_rust": "苹果锈病",
    "Apple_healthy": "苹果健康",
    "Cherry_(including_sour)_Powdery_mildew": "樱桃白粉病",
    "Cherry_(including_sour)_healthy": "樱桃健康",
    "Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot": "玉米灰斑病",
    "Corn_(maize)_Common_rust_": "玉米锈病",
    "Corn_(maize)_Northern_Leaf_Blight": "玉米大斑病",
    "Corn_(maize)_healthy": "玉米健康",
    "Grape_Black_rot": "葡萄黑腐病",
    "Grape_Esca_(Black_Measles)": "葡萄埃斯卡病",
    "Grape_Leaf_blight_(Isariopsis_Leaf_Spot)": "葡萄叶枯病",
    "Grape_healthy": "葡萄健康",
    "Orange_Haunglongbing_(Citrus_greening)": "柑橘黄龙病",
    "Peach_Bacterial_spot": "桃细菌性斑点病",
    "Peach_healthy": "桃健康",
    "Pepper,_bell_Bacterial_spot": "辣椒细菌性斑点病",
    "Pepper,_bell_healthy": "辣椒健康",
    "Potato_Early_blight": "马铃薯早疫病",
    "Potato_Late_blight": "马铃薯晚疫病",
    "Potato_healthy": "马铃薯健康",
    "Rice_Bacterial leaf blight": "水稻白叶枯病",
    "Rice_Brown_Spot": "水稻褐斑病",
    "Rice_Healthy": "水稻健康",
    "Rice_Leaf smut": "水稻叶黑粉病",
    "Rice_Leaf_Blast": "水稻稻瘟病",
    "Rice_Neck_Blast": "水稻穗颈瘟",
    "Squash_Powdery_mildew": "南瓜白粉病",
    "Strawberry_Leaf_scorch": "草莓叶焦病",
    "Strawberry_healthy": "草莓健康",
    "Tomato_Bacterial_spot": "番茄细菌性斑点病",
    "Tomato_Early_blight": "番茄早疫病",
    "Tomato_Late_blight": "番茄晚疫病",
    "Tomato_Leaf_Mold": "番茄叶霉病",
    "Tomato_Septoria_leaf_spot": "番茄斑枯病",
    "Tomato_Spider_mites Two-spotted_spider_mite": "番茄红蜘蛛危害",
    "Tomato_Target_Spot": "番茄靶斑病",
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus": "番茄黄化曲叶病毒病",
    "Tomato_Tomato_mosaic_virus": "番茄花叶病毒病",
    "Tomato_healthy": "番茄健康",
    "Wheat_healthy": "小麦健康",
    "Wheat_septoria": "小麦叶枯病",
    "Wheat_stripe_rust": "小麦条锈病",
}

# 作物 → 该作物下的病害列表
CROP_DISEASES = {
    "苹果": ["苹果黑星病", "苹果黑腐病", "苹果锈病"],
    "樱桃": ["樱桃白粉病"],
    "玉米": ["玉米灰斑病", "玉米锈病", "玉米大斑病"],
    "葡萄": ["葡萄黑腐病", "葡萄埃斯卡病", "葡萄叶枯病"],
    "柑橘": ["柑橘黄龙病"],
    "桃": ["桃细菌性斑点病"],
    "辣椒": ["辣椒细菌性斑点病"],
    "马铃薯": ["马铃薯早疫病", "马铃薯晚疫病"],
    "水稻": ["水稻白叶枯病", "水稻褐斑病", "水稻叶黑粉病", "水稻稻瘟病", "水稻穗颈瘟"],
    "南瓜": ["南瓜白粉病"],
    "草莓": ["草莓叶焦病"],
    "番茄": ["番茄细菌性斑点病", "番茄早疫病", "番茄晚疫病", "番茄叶霉病",
             "番茄斑枯病", "番茄红蜘蛛危害", "番茄靶斑病", "番茄黄化曲叶病毒病",
             "番茄花叶病毒病"],
    "小麦": ["小麦叶枯病", "小麦条锈病"],
}

CROP_CLASS_PREFIXES = {
    "苹果": ("Apple_",),
    "樱桃": ("Cherry_",),
    "玉米": ("Corn_(maize)_",),
    "葡萄": ("Grape_",),
    "柑橘": ("Orange_",),
    "桃": ("Peach_",),
    "辣椒": ("Pepper,_bell_",),
    "马铃薯": ("Potato_",),
    "水稻": ("Rice_",),
    "南瓜": ("Squash_",),
    "草莓": ("Strawberry_",),
    "番茄": ("Tomato_",),
    "小麦": ("Wheat_",),
}

CROP_ALIASES = {
    "土豆": "马铃薯",
    "马铃薯": "马铃薯",
    "西红柿": "番茄",
    "番茄": "番茄",
    "玉蜀黍": "玉米",
}

CROP_EN_ALIASES = {
    "apple": "苹果",
    "cherry": "樱桃",
    "corn": "玉米",
    "maize": "玉米",
    "grape": "葡萄",
    "orange": "柑橘",
    "citrus": "柑橘",
    "peach": "桃",
    "pepper": "辣椒",
    "bell pepper": "辣椒",
    "potato": "马铃薯",
    "rice": "水稻",
    "squash": "南瓜",
    "strawberry": "草莓",
    "tomato": "番茄",
    "wheat": "小麦",
}


def normalize_crop_name(crop_name: str) -> str:
    """统一前端传入的作物名称，便于按作物限制类别。"""
    crop_name = (crop_name or "").strip()
    if crop_name in CROP_ALIASES:
        return CROP_ALIASES[crop_name]
    return CROP_EN_ALIASES.get(crop_name.lower(), crop_name)


def get_crop_class_ids(crop_name: str) -> list:
    """返回某个作物在 44 类模型中对应的类别 ID，包含健康类别。"""
    normalized = normalize_crop_name(crop_name)
    prefixes = CROP_CLASS_PREFIXES.get(normalized)
    if not prefixes:
        return []

    return [
        idx for idx, en_name in enumerate(CLASS_NAMES)
        if any(en_name.startswith(prefix) for prefix in prefixes)
    ]

# ==================== 治疗建议 ====================
TREATMENT_MAP = {
    "苹果黑星病": [
        "及时清除病叶、病果，集中深埋或烧毁",
        "发病初期喷洒苯醚甲环唑（10%水分散粒剂1500倍液）或吡唑醚菌酯",
        "加强果园通风透光，合理修剪内膛枝条",
        "春季萌芽前喷施波尔多液（1:2:200）进行预防",
    ],
    "苹果黑腐病": [
        "剪除病枝、病果，彻底清除侵染源",
        "喷洒甲基硫菌灵（70%可湿性粉剂800倍液）或多菌灵防治",
        "果实套袋，减少病原菌侵染机会",
        "采收后彻底清园，减少越冬菌源",
    ],
    "苹果锈病": [
        "清除果园周围5公里内的桧柏等转主寄主",
        "发芽前喷洒三唑酮（20%乳油2000倍液）进行保护",
        "发病期喷洒戊唑醇或苯醚甲环唑",
        "加强肥水管理，增强树势，提高抗病能力",
    ],
    "番茄早疫病": [
        "与非茄科作物轮作2-3年，避免连作",
        "发病初期喷洒代森锰锌（80%可湿性粉剂600倍液）或嘧菌酯",
        "及时摘除下部老叶、病叶，减少病原",
        "采用高畦栽培，加强田间通风透光",
    ],
    "番茄晚疫病": [
        "及时发现并清除中心病株，带出田外深埋",
        "喷洒霜脲·锰锌（72%可湿性粉剂600倍液）或氟菌·霜霉威",
        "控制田间湿度，避免大水漫灌，采用滴灌",
        "选用抗病品种（如毛粉802），合理密植",
    ],
    "番茄叶霉病": [
        "加强通风换气，降低棚内相对湿度至80%以下",
        "喷洒嘧霉胺（40%悬浮剂1000倍液）或腐霉利",
        "合理施用氮肥，增施磷钾肥，提高植株抗性",
        "及时摘除病叶并带出田间集中处理",
    ],
    "番茄细菌性斑点病": [
        "选用无病种子，播种前用55℃温水浸种30分钟",
        "发病初期喷洒氢氧化铜（77%可湿性粉剂500倍液）",
        "与非茄科作物轮作，加强田间管理",
        "避免在叶片有露水时进行农事操作",
    ],
    "番茄黄化曲叶病毒病": [
        "防治烟粉虱等传毒媒介昆虫（悬挂黄色粘虫板）",
        "选用抗病毒品种（如浙粉702等抗TY品种）",
        "育苗期使用40-60目防虫网隔离",
        "发病初期喷洒盐酸吗啉胍（20%可湿性粉剂500倍液）",
    ],
    "番茄花叶病毒病": [
        "选用抗病品种，使用无毒种子",
        "操作前用肥皂水洗手，避免接触传毒",
        "及时拔除病株，减少田间毒源",
        "防治蚜虫，使用银灰色地膜驱避蚜虫",
    ],
    "番茄斑枯病": [
        "清除田间病残体，减少初侵染来源",
        "发病初期喷洒苯醚甲环唑或嘧菌酯",
        "合理密植，保持田间通风透光",
        "实行2年以上轮作",
    ],
    "番茄靶斑病": [
        "选用抗病品种，合理密植",
        "发病初期喷洒苯醚甲环唑或吡唑醚菌酯",
        "加强田间管理，增施有机肥",
        "及时清除病叶、病果",
    ],
    "番茄红蜘蛛危害": [
        "保护和利用天敌（如捕食螨、草蛉等）",
        "喷洒阿维菌素（1.8%乳油3000倍液）或哒螨灵",
        "保持田间湿度，红蜘蛛在干燥条件下繁殖快",
        "清除田间杂草，减少虫源",
    ],
    "玉米大斑病": [
        "选用抗病品种（如郑单958等抗大斑病品种）",
        "合理施肥，增施磷钾肥，避免偏施氮肥",
        "发病初期喷洒苯醚甲环唑（10%水分散粒剂1500倍液）",
        "收获后彻底清理田间病残体，深耕翻埋",
    ],
    "玉米锈病": [
        "选用抗锈病品种",
        "发病初期喷洒三唑酮（20%乳油2000倍液）或戊唑醇",
        "合理密植，改善田间通风条件",
        "及时清除田间杂草和自生苗",
    ],
    "玉米灰斑病": [
        "选用抗病品种，合理密植",
        "发病初期喷洒苯醚甲环唑或吡唑醚菌酯",
        "实行轮作，减少土壤中病原菌积累",
        "加强田间管理，及时排除积水",
    ],
    "马铃薯早疫病": [
        "选用无病种薯，切块时严格消毒",
        "加强水肥管理，增施钾肥提高抗病力",
        "发病初期喷洒代森锰锌（80%可湿性粉剂600倍液）或嘧菌酯",
        "收获时避免薯块损伤，入库前严格挑选",
    ],
    "马铃薯晚疫病": [
        "选用抗病品种和无病种薯",
        "发现中心病株立即拔除并深埋",
        "喷洒霜脲·锰锌或氟啶胺（50%悬浮剂2000倍液）",
        "合理灌溉，降低田间湿度，避免傍晚浇水",
    ],
    "葡萄黑腐病": [
        "冬季清园，彻底剪除病枝、病果并烧毁",
        "发芽前喷施3-5波美度石硫合剂",
        "花后喷洒苯醚甲环唑或吡唑醚菌酯",
        "果实套袋，有效减少病原菌侵染",
    ],
    "葡萄埃斯卡病": [
        "加强果园管理，增强树势",
        "修剪时避免大伤口，剪后涂抹伤口保护剂",
        "发现病株及时挖除，防止扩散",
        "合理灌溉，避免田间积水",
    ],
    "葡萄叶枯病": [
        "清除病叶、病枝，减少病原",
        "喷洒多菌灵（50%可湿性粉剂800倍液）或甲基硫菌灵",
        "加强果园通风透光，合理修剪",
        "合理施肥，避免偏施氮肥",
    ],
    "草莓叶焦病": [
        "及时摘除病叶、老叶，集中处理",
        "合理灌溉，避免叶片长期处于湿润状态",
        "喷洒嘧菌酯（25%悬浮剂1500倍液）或吡唑醚菌酯",
        "加强通风，降低棚内湿度，采用地膜覆盖",
    ],
    "柑橘黄龙病": [
        "及时挖除并销毁病树，切断传播源",
        "狠抓柑橘木虱防治（传毒媒介），统一放梢",
        "使用无病苗木建园，不从病区引种",
        "加强果园巡查，发现病树立即挖除",
    ],
    "辣椒细菌性斑点病": [
        "种子消毒：55℃温水浸种20分钟或1%硫酸铜浸种5分钟",
        "合理轮作，避免与茄科作物连作",
        "发病初期喷洒氢氧化铜（77%可湿性粉剂500倍液）或春雷霉素",
        "加强通风，降低田间湿度，采用高畦栽培",
    ],
    "桃细菌性斑点病": [
        "冬季清园，彻底剪除病枝并烧毁",
        "萌芽前喷洒波尔多液（1:1:100）或石硫合剂",
        "发病初期喷洒农用链霉素或氢氧化铜",
        "加强果园管理，增施有机肥，增强树势",
    ],
    "水稻稻瘟病": [
        "选用抗稻瘟病品种（如Y两优系列等）",
        "合理施肥，避免偏施氮肥，增施硅肥",
        "发病初期喷洒三环唑（75%可湿性粉剂1500倍液）或稻瘟灵",
        "种子消毒处理（25%咪鲜胺2000倍液浸种48小时）",
    ],
    "水稻白叶枯病": [
        "选用抗病品种，种子消毒处理（强氯精500倍液浸种24小时）",
        "发病初期喷洒噻菌铜（20%悬浮剂500倍液）或叶枯唑",
        "加强水肥管理，避免深水灌溉和串灌漫灌",
        "及时清除田间病残体，合理施用氮肥，增施磷钾肥",
    ],
    "水稻叶黑粉病": [
        "选用抗病品种，合理密植，改善通风条件",
        "种子处理：25%咪鲜胺2000倍液浸种48小时",
        "发病初期喷洒三唑酮（20%乳油2000倍液）或戊唑醇",
        "收获后深翻土壤，减少田间菌源积累",
    ],
    "水稻褐斑病": [
        "合理施肥，避免缺钾和偏施氮肥",
        "深耕改土，增施有机肥和磷钾肥",
        "发病初期喷洒苯醚甲环唑或嘧菌酯",
        "合理灌溉，避免长期深水灌溉",
    ],
    "水稻穗颈瘟": [
        "孕穗末期至齐穗期是防治关键时期，喷洒三环唑预防",
        "发病初期喷洒稻瘟灵（40%乳油1000倍液）或吡唑醚菌酯",
        "合理施肥，避免偏施氮肥，增施硅钾肥增强茎秆抗性",
        "选用抗穗颈瘟品种（如Y两优系列、深两优系列等）",
    ],
    "小麦叶枯病": [
        "选用抗病品种，合理轮作倒茬",
        "发病初期喷洒苯醚甲环唑（10%水分散粒剂1500倍液）或嘧菌酯",
        "合理施肥，增施磷钾肥，提高植株抗病能力",
        "收获后彻底清理田间病残体，深耕翻埋",
    ],
    "小麦条锈病": [
        "选用抗条锈病品种，加强监测预警",
        "早春及时喷洒三唑酮或戊唑醇防治",
        "合理密植，改善田间通风透光条件",
        "科学施肥，避免偏施氮肥",
    ],
    "樱桃白粉病": [
        "合理修剪，改善树冠通风透光",
        "发芽前喷洒5波美度石硫合剂",
        "发病初期喷洒三唑酮或嘧菌酯",
        "加强肥水管理，增强树势",
    ],
    "南瓜白粉病": [
        "合理密植，加强通风透光",
        "发病初期喷洒嘧菌酯或吡唑醚菌酯",
        "增施磷钾肥，提高植株抗性",
        "及时清除病叶，集中处理",
    ],
}

DEFAULT_TREATMENT = [
    "及时清除病叶、病株，减少病原传播",
    "加强田间管理，合理施肥灌溉",
    "咨询当地农业技术人员获取针对性防治方案",
    "注意观察病情发展，适时喷洒保护性杀菌剂",
]

HEALTHY_TREATMENT = [
    "植株生长健康，继续保持良好的田间管理措施",
    "定期巡查田间，及早发现潜在问题",
    "合理灌溉，避免土壤过干或过湿",
    "科学施肥，保证氮磷钾均衡供应",
]

# ==================== 深度学习预测 ====================

def predict_deep(image: Image.Image, top_k: int = 5, allowed_class_ids: list = None):
    """使用 ResNet 模型预测；allowed_class_ids 用于按前端选择的作物限制候选类别。"""
    global model, MODEL_READY
    if not MODEL_READY or model is None:
        return []

    try:
        img_tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1)
            if allowed_class_ids:
                class_ids_tensor = torch.tensor(allowed_class_ids, dtype=torch.long, device=DEVICE)
                top_k = min(top_k, len(allowed_class_ids))
                allowed_probs = probs[:, class_ids_tensor]
                top_probs, top_positions = torch.topk(allowed_probs, top_k)
                top_indices = class_ids_tensor[top_positions]
            else:
                top_k = min(top_k, len(CLASS_NAMES))
                top_probs, top_indices = torch.topk(probs, top_k)

        results = []
        for prob, idx in zip(
            top_probs[0].cpu().numpy(),
            top_indices[0].cpu().numpy()
        ):
            en_name = CLASS_NAMES[int(idx)] if int(idx) < len(CLASS_NAMES) else f"class_{idx}"
            cn_name = EN_TO_CN.get(en_name, en_name)
            results.append((int(idx), en_name, cn_name, float(prob)))

        return results
    except Exception as e:
        logger.warning(f"DL 推理失败: {e}")
        return []


# ==================== Flask 应用 ====================
app = Flask(__name__)
CORS(app)


def get_treatments(disease_cn: str, is_healthy: bool) -> list:
    """根据病害名称获取治疗建议"""
    if is_healthy:
        return HEALTHY_TREATMENT
    return TREATMENT_MAP.get(disease_cn, DEFAULT_TREATMENT)


# ==================== API 路由 ====================

@app.route('/api/crops', methods=['GET'])
def get_crops():
    """返回支持的作物列表"""
    crops = [
        {"name": "苹果", "icon": "🍎"},
        {"name": "樱桃", "icon": "🍒"},
        {"name": "玉米", "icon": "🌽"},
        {"name": "葡萄", "icon": "🍇"},
        {"name": "柑橘", "icon": "🍊"},
        {"name": "桃", "icon": "🍑"},
        {"name": "辣椒", "icon": "🌶️"},
        {"name": "马铃薯", "icon": "🥔"},
        {"name": "水稻", "icon": "🌾"},
        {"name": "南瓜", "icon": "🎃"},
        {"name": "草莓", "icon": "🍓"},
        {"name": "番茄", "icon": "🍅"},
        {"name": "小麦", "icon": "🌾"},
    ]
    return jsonify({"crops": crops, "total": len(crops)})


@app.route('/api/detect', methods=['POST'])
def detect():
    """
    病害识别接口
    接收: multipart/form-data
      - image: 叶片图片文件
      - crop: 农作物名称
    返回: JSON
    """
    if 'image' not in request.files:
        return jsonify({"error": "请上传图片文件"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "请选择图片文件"}), 400

    allowed_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_ext:
        return jsonify({"error": f"不支持的图片格式: {ext}，请使用 JPG/PNG/BMP/WEBP"}), 400

    file.seek(0, 2)
    size = file.tell()
    file.seek(0)
    if size > 10 * 1024 * 1024:
        return jsonify({"error": "图片文件过大，请上传 10MB 以内的图片"}), 400

    crop_name = request.form.get('crop', '').strip() or "番茄"
    image_bytes = file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # ===== 深度学习识别 =====
        crop_name = normalize_crop_name(crop_name)
        allowed_class_ids = get_crop_class_ids(crop_name)

        top_predictions = predict_deep(
            image,
            top_k=5,
            allowed_class_ids=allowed_class_ids
        )
        DL_CONFIDENCE_THRESHOLD = 0.50

        if top_predictions and top_predictions[0][2] is not None:
            best_id, best_en, best_cn, best_prob = top_predictions[0]
            is_healthy = "healthy" in best_en.lower()
            low_confidence = best_prob < DL_CONFIDENCE_THRESHOLD

            result = {
                "disease": best_cn,
                "disease_en": best_en,
                "confidence": round(best_prob, 4),
                "crop": crop_name,
                "is_healthy": is_healthy,
                "health_score": round(best_prob * 100, 1),
                "model_type": "deep_learning",
                "crop_filter_applied": bool(allowed_class_ids),
                "candidate_class_ids": allowed_class_ids,
                "low_confidence": low_confidence,
                "warning": (
                    "识别置信度较低，请确认作物选择正确，并尽量上传清晰、单片叶片、背景简单的照片。"
                    if low_confidence else ""
                ),
                "top_predictions": [
                    {
                        "disease": cn,
                        "disease_en": en,
                        "confidence": round(p, 4),
                        "class_id": class_id
                    }
                    for class_id, en, cn, p in top_predictions
                ],
                "treatments": get_treatments(best_cn, is_healthy),
            }
            logger.info(
                f"[DL] 识别: {best_cn} ({best_en}) | "
                f"选择作物: {crop_name} | "
                f"候选类别: {len(allowed_class_ids) or len(CLASS_NAMES)} | "
                f"置信度: {best_prob:.2%} | 模式: ResNet深度学习"
            )
            return jsonify(result)
        else:
            return jsonify({"error": "模型未就绪，请稍后重试"}), 503

    except Exception as e:
        logger.error(f"识别过程出错: {e}", exc_info=True)
        return jsonify({"error": f"识别失败: {str(e)}"}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        "status": "running",
        "model": "ResNet18 44类分类模型",
        "device": str(DEVICE),
        "diseases": len(CLASS_NAMES),
        "crops": len(CROP_DISEASES),
        "model_ready": MODEL_READY,
    })


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "接口不存在"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "服务器内部错误"}), 500


# ==================== 启动 ====================
if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("🌾 农作物病害智能识别系统")
    logger.info(f"   模型: ResNet18 44类分类")
    logger.info(f"   设备: {DEVICE}")
    logger.info(f"   后端地址: http://127.0.0.1:5000")
    logger.info(f"   前端页面: 在浏览器中打开 index.html")
    logger.info("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False)
