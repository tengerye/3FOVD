# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module()
class VOCDataset(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        'classes':
        ('scooter_Hero_Maestro', 'car_TataMotors_Zest', 'motorcycle_Hero_Passion', 'car_TataMotors_Indica',
        'car_Chevrolet_Spark', 'car_Nissan_Terrano', 'scooter_Suzuki_Swish', 'motorcycle_Honda_KarizmaZMR',
        'autorickshaw_Piaggio', 'car_MarutiSuzuki_Swift', 'car_Nissan_Micra', 'motorcycle_TVS_StarCityPlus',
        'truck_BharatBenz', 'car_Hyundai_Eon', 'motorcycle_RoyalEnfield_Thunderbird350', 'car_Honda_Civic',
        'car_Hyundai_Xcent', 'car_Renault_Duster', 'car_TataMotors_Tigor', 'car_Mercedes-Benz_E-Class',
        'motorcycle_TVS_ExcelHeavyDuty', 'car_TataMotors_Hexa', 'scooter_Vespa_VXL125', 'car_Mercedes-Benz_A-Class',
        'car_Honda_Amaze', 'car_MarutiSuzuki_Celerio', 'scooter_Honda_Activa', 'truck_Eicher', 'scooter_TVS_Wego',
        'car_MarutiSuzuki_Ciaz', 'car_Honda_Accord', 'car_Chevrolet_Tavera', 'motorcycle_Hero_PassionPlus',
        'car_Ford_Ikon', 'motorcycle_Yamaha_Fazer', 'motorcycle_Bajaj_PulsarRS200', 'car_Jeep_Compass', 'car_Bmw_X1',
        'car_MarutiSuzuki_Dzire', 'motorcycle_Bajaj_Pulsar200', 'motorcycle_Mahindra_Centuro',
        'motorcycle_RoyalEnfield_Classic350', 'motorcycle_Yamaha_R15', 'scooter_Honda_Dio', 'car_Mahindra_Scorpio',
        'car_Audi_A3', 'car_MarutiSuzuki_WagonR', 'car_MarutiSuzuki_Baleno', 'motorcycle_Hero_HFDeluxe',
        'motorcycle_TVS_ApacheRTR200', 'car_Jeep_Wrangler', 'motorcycle_Bajaj_Discover', 'autorickshaw_Others',
        'car_Volvo_Xc60', 'truck_AshokLeyland', 'motorcycle_Yamaha_FZS-FI', 'car_Skoda_Rapid', 'scooter_TVS_Jupiter',
        'car_Chevrolet_Enjoy', 'motorcycle_Others', 'car_Skoda_Superb', 'motorcycle_HeroHonda_SplendorNXG',
        'motorcycle_Bajaj_PulsarNS200', 'motorcycle_TVS_ApacheRTR160', 'motorcycle_Honda_Karizma',
        'motorcycle_Hero_Hunk', 'motorcycle_KTM_Duke', 'scooter_Suzuki_Burgman', 'car_Toyota_Fortuner',
        'motorcycle_Covered', 'car_MarutiSuzuki_Ertiga', 'car_Mahindra_Bolero',
        'motorcycle_RoyalEnfield_Thunderbird350X', 'car_Mahindra_Xylo', 'car_Chevrolet_Cruze',
        'motorcycle_Yamaha_RX100', 'motorcycle_Honda_CBTwister', 'car_Volkswagen_Polo', 'motorcycle_Bajaj_Discover100',
        'car_MarutiSuzuki_Zen', 'car_Mercedes-Benz_S-Class', 'motorcycle_Bajaj_Discover125', 'car_Hyundai_I20',
        'scooter_Honda_Grazia', 'car_Honda_Wr-V', 'motorcycle_TVS_Sport', 'car_Volkswagen_Vento',
        'car_MarutiSuzuki_SX4', 'scooter_Hero_Duet', 'scooter_TVS_Streak', 'scooter_Hero_Pleasure', 'car_Ford_Fiesta',
        'car_Honda_Brio', 'autorickshaw_Bajaj', 'car_Mercedes-Benz_C-Class', 'scooter_Mahindra_Gusto',
        'scooter_Suzuki_Access', 'motorcycle_TVS_XL100', 'motorcycle_Honda_StunnerCBF', 'car_MarutiSuzuki_Esteem2000',
        'motorcycle_Hero_PassionPro', 'car_MarutiSuzuki_S-Cross', 'car_TataMotors_Nano', 'motorcycle_Hero_Glamour',
        'motorcycle_Suzuki_Samurai', 'car_Bmw_2-Series-220d', 'car_Volkswagen_Ameo', 'car_MarutiSuzuki_AltoK10',
        'car_Skoda_Fabia', 'car_Toyota_Corolla', 'motorcycle_Hero_XPulse200', 'motorcycle_RoyalEnfield_Bullet350',
        'car_Chevrolet_Beat', 'car_Renault_Lodgy', 'car_Hyundai_I10', 'car_Audi_Q3', 'car_Mahindra_XUV500',
        'car_Honda_Brv', 'car_Bmw_3-Series', 'motorcycle_Suzuki_Slingshot', 'car_Chevrolet_Aveo',
        'motorcycle_RoyalEnfield_Interceptor650', 'car_Others', 'car_Mahindra_Reva', 'motorcycle_Bajaj_Avenger',
        'car_Toyota_Qualis', 'motorcycle_Bajaj_Discover135', 'autorickshaw_TVS', 'car_Hindustan_Ambassador',
        'motorcycle_Suzuki_Gixxer', 'car_Ford_Ecosport', 'car_Hyundai_Santro', 'car_MarutiSuzuki_Ignis',
        'motorcycle_RoyalEnfield_Classic500', 'motorcycle_RoyalEnfield_Meteor350', 'car_MarutiSuzuki_VitaraBrezza',
        'car_Fiat_PuntoEvo', 'car_Toyota_EtiosLiva', 'motorcycle_Yamaha_FZ25', 'motorcycle_Bajaj_Platina',
        'motorcycle_Yamaha_Crux', 'car_Ford_Everest', 'motorcycle_Honda_Shine', 'scooter_Others', 'car_Ford_Figo',
        'motorcycle_Bajaj_CT100', 'car_Mercedes-Benz_Gla-Class', 'car_Mercedes-Benz_AmgGt4-DoorCoupe',
        'scooter_TVS_Zest', 'autorickshaw_Mahindra', 'car_Fiat_Linea', 'motorcycle_Bajaj_Pulsar220F',
        'car_MarutiSuzuki_Omni', 'car_Renault_Kwid', 'car_Honda_Cr-V', 'motorcycle_Bajaj_V15', 'scooter_Bajaj_Chetak',
        'car_MarutiSuzuki_Ritz', 'car_Mahindra_Verito', 'motorcycle_RoyalEnfield_ContinentalGT650',
        'motorcycle_Bajaj_Discover110', 'bus', 'motorcycle_RoyalEnfield_Bullet500', 'truck_Others', 'car_Renault_Scala',
        'car_Toyota_Innova', 'car_Nissan_Sunny', 'car_Toyota_Etios', 'car_Mahindra_TUV300',
        'motorcycle_Hero_SuperSplendor', 'car_Covered', 'car_MarutiSuzuki_1000', 'car_TataMotors_Nexon',
        'car_Hyundai_Creta', 'mini-bus_Others', 'motorcycle_TVS_Excel100', 'motorcycle_Bajaj_V12',
        'car_TataMotors_Sumo', 'car_Honda_Jazz', 'car_Hyundai_Accent', 'scooter_Yamaha_RayZR', 'car_Volkswagen_Jetta',
        'autorickshaw_Covered', 'motorcycle_Yamaha_Libero', 'car_Skoda_Octavia', 'car_MarutiSuzuki_Alto800',
        'scooter_Yamaha_Fascino125', 'autorickshaw_Atul', 'motorcycle_Honda_Unicorn', 'car_Force_TraxToofan',
        'car_Mercedes-Benz_G-Class', 'car_Ford_EcoSportTitanium', 'truck_Mahindra', 'scooter_TVS_Pep',
        'car_TataMotors_Indigo', 'car_MarutiSuzuki_Eeco', 'scooter_TVS_Ntorq', 'truck_Tata', 'truck_SML',
        'car_Volvo_Xc40', 'motorcycle_Bajaj_Pulsar180', 'car_Renault_Logan', 'motorcycle_HeroHonda_CBZ',
        'car_TataMotors_Tiago', 'motorcycle_Honda_CBHornet160R', 'car_Hyundai_Verna', 'car_Ford_Aspire',
        'motorcycle_Honda_SP125', 'motorcycle_Hero_Splendor', 'motorcycle_Yamaha_FZ-V3', 'car_TataMotors_Safari',
        'scooter_Honda_Aviator', 'car_Honda_City', 'motorcycle_Bajaj_Pulsar150', 'motorcycle_TVS_Victor'),
        # ('Volkswagen_Jetta', 'Audi_Q5L', 'Modern_HappyMovement', 'Kia_K5', 'Mazda_5', 'Mg_ZS', 'Toyota_Rayling','Jianghuai_RuifengM3', 'Toyota_Corolla', 'Toyota_Vios', 'WeiPai_VV7', 'Ford_Explorer', 'Toyota_ViosFS','Ford_Focus', 'Lexus_ES', 'Honda_Accord', 'Gac_TrumpchiGS8', 'Honda_Civic', 'Volkswagen_ID.4CROZZ','Zhonghua_V3', 'BMW_5Series', 'Ideal_ONE', 'Toyota_Prado', 'BMW_1series', 'Buick_GL8', 'Modern_Elantra','Byd_TangDM', 'Haver_M6', 'Honda_CR-V', 'BeiqiWeiwang_306EV', 'Byd_Song', 'Baojun_310W', 'other','Lexus_RX', 'Chevrolet_Chariot', 'Modern_YueNa', 'Wuling_Hongguang', 'Changan_Escape', 'Nissan_Sylphy','Audi_A4', 'Suzuki_KaiYue', 'EastWind_FengshenE70', 'Benz_Aclass', 'Honda_InshiPie', 'Faw_PentiumB90','Idea_S1', 'Volkswagen_Sagitta', 'GoldCup_Kreis', 'Modern_Lead', 'Volvo_XC40', 'Mercedes_GLA', 'Kia_Run','Modern_ix35', 'Nissan_Versa', 'Honda_Odyssey', 'Peugeot_508', 'Toyota_Highlander', 'Volkswagen_Tiguan','Honda_BinZhi', 'Toyota_YARiSLtoDazzle', 'Chevrolet_Cruze', 'Jeep_FreeLight', 'Mercedes-benzE-class','Suzuki_TianyuSX4', 'Audi_Q5', 'Mitsubishi_Oland', 'Toyota_Camry', 'Ford_Win', 'Volkswagen_Bora','EastWind_Jingyi', 'Toyota_Corolla', 'Nissan_Qijun', 'Zhonghua_Zunchi', 'Mercedes_Benz','Honda_OzzieDazzle', 'Nissan_Bluebird', 'Toyota_RAV4Rongfang', 'Buick_Regal', 'Audi_A6', 'Honda_XR-V','Buick_Envision', 'Harvard_H6', 'Skoda_Octavia', 'Buick_Excelle', 'Honda_Elisen', 'Audi_Q3','Volkswagen_Magotan', 'unknown', 'Faw_PentiumX80', 'Emgrand_GS', 'Suzuki_Fronuis', 'Chery_Tigo5','Geely_Emgrand', 'Benz_C', 'BMW_3series', 'Suzuki_Swift', 'Mitsubishi_Pajero', 'Audi_A8', 'Dodge_Kubo','Chevrolet_Copaci', 'China_JunjieFRV', 'Doublering_Smallaristocrat', 'Toyota_Alesen','Mitsubishi_PowershineASX', 'Citroen_Sega', 'Buick_Encore', 'Ford_SharpWorld', 'Modern_Sonata','Hyundai_Shengda', 'LandRover_Godwalker', 'Mazda_3', 'Byd_F0', 'Nissan_Qashqai', 'Nio_ES8','Modern_Veloster', 'Byd_F3', 'Cadillac_SRX', 'Chery_Fengyun2', 'Changan_CS35PLUS', 'Harvard_H2','Volkswagen_Charan', 'Faw_XialiN5', 'Jeep_Guide', 'Volkswagen_JettaVA3', 'BaoJun_560', 'Cadillac_XT4','Volkswagen_POLO', 'Cheetah_Q6', 'Cheeta_CS10', 'Faw_TrumpchiGA8', 'Volkswagen_Golf', 'Skoda_Sharp','Audi_RS7', 'Kia_Rio', 'Benz_Cclass', 'Wuling_Glory', 'Geely_GX7', 'Kia_Soul', 'Changan_YueXiangV3','Toyota_CorollaSharpening', 'Faw_TrumpchiM8', 'Weibrand_VV6', 'Kia_K2', 'Mass_Tuon', 'Changan_CS75','Volkswagen_Santana', 'Toyota_YARiSLtoenjoy', 'BMW_X1', 'BMW_X6', 'Seahorse_FormilaiF5', 'Modern_LangMove','Volvo_S90', 'Modern_ShengdaClassic', 'Buick_Inlong', 'Volvo_XCClassic', 'Byd_S6', 'BMW_X3', 'Kia_Furedi','Porsche_Panamera', 'Citroen_Triumph', 'Mazda_Atz', 'LandRover_Discovery', 'Mitsubishi_Outlander','GoldenCup_SeaLionKing', 'DongfengKaiChen_KaiChenD60', 'GoldCup_SeaLion', 'Volkswagen_Lavida','Volvo_XC90', 'TheGreatWall_Gun', 'Mitsubishi_Unknown', 'Changan_AuchanX70A', 'Toyota_C-HR', 'Haver_H5','Lexus_NX', 'Peugeot_408', 'Mazda_6', 'BaoJun_310', 'Zhonghua_V5', 'Honda_UR-V', 'Flag_ds7', 'Honda_Sidi','BaoJun_730', 'Audi_A3', 'Sealion_X30', 'LANMap_Dreamers', 'Chevrolet_Lezzi', 'Chevrolet_Cowards','Modern_Swiss', 'Volkswagen_JettaVS5', 'Ford_Forus', 'Toyota_Linfang', 'Renault_Correga', 'BaoJun_510','Byd_Qin', 'Dongfeng_Lingzhi', 'Public_Tanyue', 'BMW_X5', 'BaoJun_530', 'Chery_Tigo3x', 'Faw_Weizhi','LandWind_LandWindX7', 'Zotye_DamaiX7', 'Nissan_Teana', 'Cadillac_CTS', 'Mercedes_Benz', 'Haver_h1','Honda_Jed', 'Nissan_Sunshine', 'Mazda_3Enxela', 'Hyundai_Tucson', 'Beijing_X3', 'Audi_A6L', 'Buick_G6L','Dodge_Coolway', 'Chery_Tigo5X', 'Roewe_RX5MAX', 'Audi_A4L', 'Nissan_Lixil', 'Ford_Mondeo','Chevrolet_Malibu', 'EastWind_Scenery580', 'Honda_CrownRoad', 'Byd_S7', 'Kia_K3', 'Volkswagen_Touran','TheGreatWall_M2', 'Mazda_2', 'Volkswagen_TouranL', 'Mazda_CX-4', 'Volkswagen_Tuyue', 'Geely_BoYue','Gac_TrumpCHIGS5', 'Geely_VisionX6', 'Chery_RiichM1', 'Byd_QinPLUSnewenergy', 'Ford_Collar','Honda_HaoYing', 'Changan_CS55PLUS', 'Peugeot_307', 'Wuling_Stars', 'Hafei_Horseracing', 'Chery_E3','Changan_CS75PLUS', 'Roewe_RX3', 'Harvard_F7', 'Peugeot_2008', 'Changan_YueXiang', 'Peugeot_3008','TheGreatWall_C30', 'WeiBrand_Tank300', 'Byd_QinPlusDMi', 'Byd_F3DM', 'Modern_Namedrive', 'Harvard_H2S','Faw_TrumpchiGS3', 'Benz_Bclass', 'EastWind_JingyiX3', 'Buick_Lacrosse', 'Landwind_X8','Wuling_HongguangS3', 'EastWind_JingyiX5', 'Mg_6', 'Mercedes-benz_Rclass', 'Oriental_Scenery580','Honda_FengFan', 'Volvo_S40', 'Citroen_C4L', 'LandRover_DiscoveryWalk', 'Chinese_H3', 'Audi_Q7','EastWind_Scenery', 'Benz_GLC', 'RedFlag_E-QM5', 'Lincoln_MKC', 'Byd_SongPLUSDMi', 'Lexus_CT', 'Byd_HanDM','Baic_ViwangM20', 'Volkswagen_Phaeton', 'Mazda_CX-5', 'Honda_Fit', 'Benz_GLE', 'Tesla_ModelY','Volvo_XC60', 'Baic_Viwang306', 'Changan_YueXiangV7', 'Honda_Lampai', 'Toyota_Gravia', 'Volkswagen_Passat','Fukuda_SceneryG5', 'Buick_Veran', 'Volkswagen_TiguanL', 'Modern_Rena', 'GoldCup_ZhishangS30','Jianghuai_RuifengS3', 'Infiniti_QX50', 'Roewe_RX8', 'Changan_CS35', 'Chevrolet_Adventurer', 'Geely_ICON','Peugeot_301', 'Link_02', 'Geely_Vision', 'Chery_Tigo3', 'Seahorse_FortuneStar', 'Nissan_Edda','Nissan_Tule', 'Changan_CS15', 'Geely_VisionX3', 'Toyota_Willanda', 'Lifan_XuanLang','Volkswagen_JettaVS7', 'Subaru_Forester', 'Mitsubishi_WingedGod', 'Changan_AuchanZ6', 'Chery_QQ','Nissan_Paladin', 'Toyota_HILUX', 'Jiangling_Yuhu5', 'Pentium_X80', 'EastWind_Wind330', 'JiangLing_E200n','Chevrolet_View', 'Toyota_InshiPie', 'Cheetah_BlackGoldGang', 'Redflag_HS5', 'WeiPai_VV5', 'Toyota_Reiz','Toyota_LandcoolLuze', 'Public_SongExploration', 'Volkswagen_CC', 'EastWind_popularSX6', 'Byd_E3','Ford_TheRoadShaker', 'Tesla_Modelx', 'Suzuki_Vitra', 'LandRover_RangeRoverEvoque', 'Harvard_H6Coupe','Modern_ix25', 'Citroen_Picasso', 'BMW_7Series', 'Nissan_Loulan', 'TheGreatWall_M4', 'Benz_CLS','Cyrus_AsktheworldM5', 'Citroen_C5', 'Lincoln_TheAviator', 'SaicMAXUSV80', 'Skoda_Sharp', 'Kia_K3S','Audi_Q5Sportback', 'Chevrolet_Lefeng', 'Changan_Wing-beat', 'Opel_Vista', 'GreatWall_Tank300','Toyota_Crown', 'Buick_MicroBlue6', 'Changan_UNI-K', 'Byd_QinNewEnergy', 'Volkswagen_Langxing', 'Haver_H7','Skoda_FastPie', 'Skoda_Cormick', 'Kia_SmartRun', 'Mazda_2Jinxiang', 'Peugeot_208', 'Pentium_B50','LandRover_RangeRover', 'Modern_Jed', 'Byd_Yuan', 'Chery_Arezer5', 'Citroen_Elysee', 'Changan_Benbenmini','Volkswagen_C-TREKNioCollar', 'Chery_E5', 'Geely_BinYue', 'Chery_Arezer3', 'Chery_ArezerGX', 'Kia_Kessen','Faw_TrumpchiGS4', 'Usheng_S330', 'Volkswagen_Sharp', 'Faw_TrumpchiGS5', 'Changan_AuchanA600', 'Roewe_RX5','Volkswagen_Lamdo', 'Toyota_Gori', 'Byd_Suirui', 'Dongfeng_JingyiS50', 'Modern_i30', 'TheGreatWall_C20R','Ford_Escape', 'Changan_UNI-T', 'Buick_Enclave', 'Jianghuai_Ruiwind', 'Geely_EmgrandL','Volkswagen_Driving', 'Faw_Orang', 'Faw_PentiumB70', 'LandRover_RangeRoverSport', 'Kia_Sportage','smart_forfour', 'Chery_Tigress7', 'Qoros_3', 'TheGreatWall_C50', 'Audi_A5', 'Volvo_S60','Baic_MagicSpeedS3', 'Link_03', 'Byd_Dolphin', 'Renault_Koreo', 'Mitsubishi_YiSong', 'Porsche_Macan','Roewe_Ei5', 'Modern_FrontFan', 'Toyota_Prius', 'MINI_Cooper', 'Modern_Fex', 'Audi_A8L','Chevrolet_Trailblazer', 'Volkswagen_Phaeon', 'Volvo_v60', 'Ford_Rayworld', 'China_Junjie', 'Mazda_8','Faw_PentiumB50', 'Jietu_X70PLUS', 'Volkswagen_T-ROCexploration', 'Ford_Fiesta', 'Modern_SantaFe','Geely_XingyueL', 'Byd_G3', 'Hyundai_Equus', 'Chinese_H530', 'ChanganUNI-V', 'Gili_Diamond','Chevrolet_MusicRV', 'Roewe_i5', 'Volkswagen_TanyueGTE', 'Baojun_RS-5', 'Toyota_Fortuner', 'Roewe_350','Dongfeng_WinwindS500', 'Haval_F7x', 'Toyota_Privia', 'Changan_Olivay', 'Toyota_Yaris'),
        # ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        #  'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        #  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'),
        # ('\u6BD4\u4E9A\u8FEA_F3', '\u51EF\u8FEA\u62C9\u514B_SRX', '\u957F\u5B89_CS35PLUS', '\u54C8\u5F17_H2', 'other', '\u5927\u4F17_\u590F\u6717',
        # '\u5947\u745E_\u98CE\u4E912', '\u5B9D\u9A8F_560', '\u4E00\u6C7D_\u590F\u5229N5', '\u5409\u666E_\u6307\u5357\u8005', '\u51EF\u8FEA\u62C9\u514B_XT4',
        # '\u5927\u4F17_\u6377\u8FBE', '\u94C3\u6728_\u96E8\u71D5', '\u5927\u4F17_POLO', '\u730E\u8C79_Q6', '\u5927\u4F17_\u5B9D\u6765', '\u730E\u8C79_CS10',
        # '\u4E00\u6C7D_\u4F20\u797AGA8', '\u65E5\u4EA7_\u900D\u5BA2', '\u5927\u4F17_\u9AD8\u5C14\u592B', '\u65AF\u79D1\u8FBE_\u6615\u9510', '\u5965\u8FEA_RS7',
        # '\u5409\u5229_\u5E1D\u8C6A', '\u4E2D\u534E_V3', '\u8D77\u4E9A_\u9510\u6B27', '\u4E94\u83F1_\u8363\u5149', '\u5954\u9A70_C\u7EA7', '\u5409\u5229_GX7',
        # '\u8D77\u4E9A_Soul', '\u957F\u5B89_\u60A6\u7FD4V3', '\u4E30\u7530_\u5361\u7F57\u62C9\u9510\u653E', '\u4E00\u6C7D_\u4F20\u797AM8', '\u73B0\u4EE3_\u9886\u52A8',
        # '\u5927\u4F17_\u6377\u8FBEVA3', '\u96EA\u4F5B\u5170_\u8D5B\u6B27', '\u9B4F\u724C_VV6', '\u8D77\u4E9A_K2', '\u5927\u4F17_\u9014\u6602', '\u957F\u5B89_CS75',
        # '\u5927\u4F17_\u6851\u5854\u7EB3', 'unknown', '\u4E30\u7530_YARiSL\u81F4\u4EAB', '\u54C8\u5F17_H6', '\u4E30\u7530_RAV4\u8363\u653E', '\u5B9D\u9A6C_X6',
        # '\u4E30\u7530_\u5361\u7F57\u62C9', '\u5B9D\u9A6C_X1', '\u798F\u7279_\u9510\u754C', '\u53CC\u73AF_\u5C0F\u8D35\u65CF', '\u96F7\u514B\u8428\u65AF_ES',
        # '\u6D77\u9A6C_\u798F\u7F8E\u6765F5', '\u5927\u4F17_\u8FC8\u817E', '\u73B0\u4EE3_\u6717\u52A8', '\u5B9D\u9A6C_1\u7CFB', '\u6C83\u5C14\u6C83_S90', '\u73B0\u4EE3_\u80DC\u8FBE\u7ECF\u5178',
        # '\u522B\u514B_\u82F1\u6717', '\u672C\u7530_CR-V', '\u6C83\u5C14\u6C83_XC Classic', '\u6BD4\u4E9A\u8FEA_S6', '\u798F\u7279_\u798F\u514B\u65AF', '\u73B0\u4EE3_\u60A6\u52A8', '\u5B9D\u9A6C_X3',
        # '\u5954\u9A70_GLA', '\u8D77\u4E9A_\u798F\u745E\u8FEA', '\u4FDD\u65F6\u6377_Panamera', '\u96EA\u94C1\u9F99_\u51EF\u65CB', '\u522B\u514B_\u51EF\u8D8A', '\u9A6C\u81EA\u8FBE_\u963F\u7279\u5179',
        # '\u96EA\u94C1\u9F99_\u4E16\u5609', '\u65E5\u4EA7_\u8F69\u9038', '\u8DEF\u864E_\u53D1\u73B0', '\u4E09\u83F1_\u6B27\u84DD\u5FB7', '\u5965\u8FEA_Q5L', '\u91D1\u676F_\u6D77\u72EE\u738B', '\u5B9D\u9A6C_3\u7CFB',
        # '\u5927\u4F17_\u901F\u817E', '\u4E1C\u98CE\u542F\u8FB0_\u542F\u8FB0D60', '\u91D1\u676F_\u6D77\u72EE', '\u65E5\u4EA7_\u9A90\u8FBE', '\u5927\u4F17_\u6717\u9038', '\u6C83\u5C14\u6C83_XC90',
        # '\u672C\u7530_\u96C5\u9601', '\u73B0\u4EE3_ix35', '\u957F\u57CE_\u70AE', '\u4E09\u83F1_\u672A\u77E5', '\u522B\u514B_\u6602\u79D1\u62C9', '\u5B9D\u9A6C_5\u7CFB', '\u4E30\u7530_RAVA\u8363\u653E',
        # '\u4E30\u7530_C-HR', '\u957F\u5B89_\u6B27\u5C1AX70A', '\u54C8\u5F17_H5', '\u96F7\u514B\u8428\u65AF_NX', '\u672C\u7530_\u827E\u529B\u7EC5', '\u4E30\u7530_\u96F7\u51CC', '\u522B\u514B_GL8',
        # '\u6807\u81F4_408', '\u9A6C\u81EA\u8FBE_6', '\u6BD4\u4E9A\u8FEA_\u5B8B', '\u54C8\u5F17_M6', '\u5B9D\u9A8F_310', '\u4E30\u7530_\u6C49\u5170\u8FBE', '\u4E2D\u534E_V5', '\u672C\u7530_UR-V',
        # '\u6807\u5FD7_ds7', '\u672C\u7530_\u601D\u8FEA', '\u5B9D\u9A8F_730', '\u5965\u8FEA_A3', '\u6D77\u72EE_X30', '\u5C9A\u56FE_\u68A6\u60F3\u5BB6', '\u96EA\u4F5B\u5170_\u4E50\u9A70',
        # '\u96EA\u4F5B\u5170_\u79D1\u6C83\u5179', '\u4E94\u83F1_\u5B8F\u5149', '\u9A6C\u81EA\u8FBE_3', '\u5965\u8FEA_A6', '\u73B0\u4EE3_\u745E\u5955', '\u5409\u666E_\u81EA\u7531\u5149',
        # '\u5927\u4F17_\u6377\u8FBEVS5', '\u798F\u7279_\u798F\u777F\u65AF', '\u4E30\u7530_\u51CC\u653E', '\u96EA\u4F5B\u5170_\u79D1\u9C81\u5179', '\u96F7\u8BFA_\u79D1\u96F7\u5609', '\u5B9D\u9A8F_510',
        # '\u672C\u7530_\u601D\u57DF', '\u6BD4\u4E9A\u8FEA_\u79E6', '\u4E1C\u98CE_\u83F1\u667A', '\u5927\u4F17_\u63A2\u5CB3', '\u672C\u7530_\u82F1\u4ED5\u6D3E', '\u5947\u745E_\u745E\u864E5', '\u5B9D\u9A6C_X5',
        # '\u5B9D\u9A8F_530', '\u5965\u8FEA_A4', '\u73B0\u4EE3_\u7D22\u7EB3\u5854', '\u5947\u745E_\u745E\u864E3X', '\u4E00\u6C7D_\u5A01\u5FD7', '\u9646\u98CE_\u9646\u98CEX7', '\u5965\u8FEA_Q3', '\u672C\u7530_XR-V',
        # '\u65E5\u4EA7_\u9014\u4E50', '\u957F\u5B89_CS15', '\u5409\u5229_\u8FDC\u666FX3', '\u4E30\u7530_\u5A01\u5170\u8FBE', '\u8D77\u4E9A_K3', '\u529B\u5E06_\u8F69\u6717', '\u5927\u4F17_\u6377\u8FBEVS7',
        # '\u798F\u7279_\u8499\u8FEA\u6B27', '\u65AF\u5DF4\u9C81_\u68EE\u6797\u4EBA', '\u672C\u7530_\u98DE\u5EA6', '\u957F\u5B89_CS55PLUS', '\u4E09\u83F1_\u7FFC\u795E', '\u4E30\u7530_\u82B1\u51A0',
        # '\u957F\u5B89_\u6B27\u5C1AZ6', '\u5947\u745E_QQ', '\u65E5\u4EA7_\u5E15\u62C9\u4E01', '\u4E30\u7530_HILUX', '\u5176\u4ED6', '\u6C5F\u94C3_\u57DF\u864E5', '\u5954\u817E_X80', '\u4E1C\u98CE_\u98CE\u5149330',
        # '\u5927\u4F17_\u9014\u5CB3', '\u6C5F\u94C3_E200N', '\u957F\u5B89_\u60A6\u7FD4', '\u672C\u7530_\u7693\u5F71', '\u96EA\u4F5B\u5170_\u666F\u7A0B', '\u4E30\u7530_\u82F1\u4ED5\u6D3E', '\u730E\u8C79_\u9ED1\u91D1\u521A',
        # '\u7EA2\u65D7_HS5', '\u5927\u4F17_\u5E15\u8428\u7279', '\u9B4F\u724C_VV5', '\u4E30\u7530_\u9510\u5FD7', '\u4E30\u7530_\u5170\u5FB7\u9177\u8DEF\u6CFD', '\u65E5\u4EA7_\u5929\u7C41', '\u65E5\u4EA7_\u9890\u8FBE',
        # '\u5954\u9A70_GLE', '\u5927\u4F17_CC', '\u5927\u4F17_\u63A2\u6B4C', '\u4E1C\u98CE_\u98CE\u884CSX6', '\u6BD4\u4E9A\u8FEA_e3', '\u5965\u8FEA_A6L', '\u5927\u4F17_\u9014\u89C2L', '\u8DEF\u864E_\u63FD\u80DC\u6781\u5149',
        # '\u6807\u81F4_307', '\u5965\u8FEAA6L', '\u73B0\u4EE3_\u9014\u80DC', '\u65AF\u67EF\u8FBE_\u660E\u9510', '\u54C8\u5F17_H6Coupe', '\u5954\u9A70_GLB', '\u73B0\u4EE3_ix25', '\u94C3\u6728_\u5929\u8BEDSX4',
        # '\u82F1\u83F2\u5C3C\u8FEA_QX50', '\u96EA\u94C1\u9F99_\u6BD5\u52A0\u7D22', '\u73B0\u4EE3_\u4F0A\u5170\u7279', '\u5B9D\u9A6C_7\u7CFB', '\u6BD4\u4E9A\u8FEA_\u79E6PlusDMi', '\u65E5\u4EA7_\u697C\u5170',
        # '\u522B\u514B_\u5A01\u6717', '\u957F\u57CE_M4', '\u5954\u9A70_CLS', '\u96EA\u94C1\u9F99_C5', '\u6797\u80AF_\u98DE\u884C\u5BB6', '\u4E0A\u6C7D\u5927\u901A_MAXUS V80', '\u8D5B\u529B\u65AF_\u95EE\u754CM5',
        # '\u5965\u8FEA_A4L', '\u65AF\u67EF\u8FBE_\u6615\u9510', '\u8D77\u4E9A_K3S', '\u672C\u7530_\u51CC\u6D3E', '\u5965\u8FEA_Q5 Sportback', '\u96EA\u4F5B\u5170_\u4E50\u4E30', '\u957F\u5B89_\u7FFC\u640F',
        # '\u957F\u5B89_\u9038\u52A8', '\u6B27\u5B9D_\u5A01\u8FBE', '\u5954\u9A70_B\u7EA7', '\u957F\u57CE_\u5766\u514B300', '\u9A6C\u81EA\u8FBE_CX-4', '\u4E30\u7530_\u7687\u51A0', '\u522B\u514B_\u5FAE\u84DD6', '\u5965\u8FEAQ3', '\u9A6C\u81EA\u8FBE_CX-5', '\u5954\u9A70_GLC', '\u65E5\u4EA7_\u9A8A\u5A01', '\u4E30\u7530_\u51EF\u7F8E\u745E', '\u957F\u5B89_UNI-K', '\u4E30\u7530_\u5A01\u9A70', '\u957F\u5B89_CS75PLUS', '\u65AF\u67EF\u8FBE_\u901F\u6D3E', '\u65AF\u67EF\u8FBE_\u67EF\u7C73\u514B', '\u672C\u7530_\u7F24\u667A', '\u9A6C\u81EA\u8FBE_3\u6602\u514B\u8D5B\u62C9', '\u54C8\u98DE_\u8D5B\u9A6C', '\u5954\u9A70_R\u7EA7', '\u73B0\u4EE3_\u950B\u8303', '\u4E30\u7530_\u666E\u9510\u65AF', '\u5927\u4F17_\u9014\u89C2', 'MINI_Cooper', '\u4E30\u7530_\u666E\u62C9\u591A', '\u96F7\u8BFA_\u79D1\u96F7\u50B2', '\u7279\u65AF\u62C9_ModelX', '\u73B0\u4EE3_\u98DE\u601D', '\u4E30\u7530_YARiSL\u81F4\u70AB', '\u9A6C\u81EA\u8FBE_2', '\u8D77\u4E9A_\u72EE\u8DD1', '\u94C3\u6728_\u7EF4\u7279\u62C9', '\u94C3\u6728_\u950B\u9A6D', '\u798F\u7279_\u64BC\u8DEF\u8005', '\u5965\u8FEA_A8L', '\u96EA\u4F5B\u5170_\u5F00\u62D3\u8005', '\u5927\u4F17_\u8F89\u6602', '\u5954\u9A70_E\u7EA7', '\u6C83\u5C14\u6C83_v60', '\u798F\u7279_\u9510\u9645', '\u4E2D\u534E_\u9A8F\u6377', '\u9A6C\u81EA\u8FBE_8', '\u6BD4\u4E9A\u8FEA_S7', '\u4E30\u7530_\u5A01\u9A70FS', '\u4E00\u6C7D_\u5954\u817EB50', '\u5927\u4F17_ID.4 CROZZ', '\u5927\u4F17_T-ROC\u63A2\u6B4C', '\u4E09\u83F1_\u52B2\u70ABASX', '\u798F\u7279_\u5609\u5E74\u534E', '\u6BD4\u4E9A\u8FEA_F3DM', '\u73B0\u4EE3_\u5723\u8FBE\u83F2', '\u5409\u5229_\u661F\u8D8AL', '\u6BD4\u4E9A\u8FEA_G3', '\u65E5\u4EA7_\u9633\u5149', '\u73B0\u4EE3_\u96C5\u79D1\u4ED5', '\u4E2D\u534E_H530', '\u798F\u7279_\u81F4\u80DC', '\u957F\u57CE_C50', '\u4E30\u7530_Fortuner', '\u5965\u8FEA_A5', '\u8363\u5A01_350', '\u4E1C\u98CE_\u98CE\u884CS500', '\u798F\u7279_\u7FFC\u864E', '\u5927\u4F17_\u63A2\u5CB3GTE', '\u5B9D\u9A8F_RS-5', '\u5947\u745E_\u745E\u864E3', '\u6BD4\u4E9A\u8FEA_\u79E6\u65B0\u80FD\u6E90', '\u54C8\u5F17_F7x', '\u65E5\u4EA7_\u5947\u9A8F', '\u4E30\u7530_\u666E\u745E\u7EF4\u4E9A',
        # '\u957F\u5B89_\u6B27\u529B\u5A01', '\u672C\u7530_\u950B\u8303', '\u4E30\u7530_\u96C5\u529B\u58EB'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157), (0, 182, 199),
                    (0, 226, 252), (182, 182, 255), (0, 0, 230), (220, 20, 60),
                    (163, 255, 0), (0, 82, 0), (3, 95, 161), (0, 80, 100),
                    (183, 130, 88)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'VOC2007' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2007'
        elif 'VOC2012' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2012'
        else:
            self._metainfo['dataset_type'] = None
