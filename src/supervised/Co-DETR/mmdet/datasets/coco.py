# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import io
import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class CocoDataset(CustomDataset):

    CLASSES = ('Fiat_Feixiang', 'Ford_Focus', 'Volkswagen_Magotan', 'Peugeot_408', 'Suzuki_KaiYue', 'Jeep_FreedomMan', 'Toyota_Corolla', 'Gac_TrumpCHIGS4', 'Baic_ViwangM30', 'Kia_Furedi', 'Volkswagen_lavida', 'Zotye_DamaiX7', 'Honda_Accord', 'BMW_5Series', 'Byd_F3', 'Nissan_Teana', 'Cadillac_cts', 'Changan_CS75', 'Changan_CS75PLUS', 'Honda_CrownRoad', 'Haver_m6', 'Hyundai_Shengda', 'AudiA4', 'Volkswagen_Jetta', 'Toyota_Elfa', 'Modern_HappyMovement', 'Toyota_RAV4Rongfang', 'Kia_Sportage', 'Mazda_6', 'other', 'Audi_a6', 'Chery_QQ', 'Jiangling_Yuhu5', 'Toyota_HILUX', 'unknown', 'Ford_Mondeo', 'Mazda_Atz', 'Mercedes-Benz_M-Class', 'Haver_h6', 'Honda_Civic', 'Lord6', 'Citroen_Elysee', 'Mazda_cx-5', 'Geely_EmgrandGL', 'Volkswagen_JettaVS5', 'Lucky_KingKong', 'Nissan_Sunshine', 'Nissan_Sylphy', 'Orient_Scenery580', 'Modern_Elantra', 'Benz_RClass', 'Nissan_Qijun', 'Citroen_C3-XR', 'Faw_XiALIN5', 'Volvo_XC90', 'Chery_Tigo3X', 'Toyota_Highlander', 'Mazda_3', 'Nissan_Qashqai', 'Byd_Song', 'Lexus_ux', 'Nissan_versa', 'BMW_X3', 'Lexus_ES', 'Volkswagen_Santana', 'Volkswagen_Golf', 'Hongq_h7', 'Roewe_EI5', 'Modern_ix35', 'Chevrolet_Sail', 'Chevrolet_Cruze', 'Volkswagen_Sagitar', 'Buick_Lacrosse', 'LandWind_X8', 'Jeep_FreeLight', 'Honda_Benzhi', 'Geely_BoYue', 'Benz_GLB', 'Honda_CR-V', 'Public_Wirin', 'Jianghuai_RuifengS3', 'Public_Bora', 'Chery_Tigo7Plus', 'Suzuki_Swift', 'Kia_K3', 'Geely_Emgrand', 'Roewe_350', 'Volkswagen_Tuang', 'RenaultCorrega', 'Dongfeng_Lingzhi', 'Baojun_510', 'Cherry_IrizerGX', 'Buick_Yinglong', 'Modern_Langdong', 'Honda_Odyssey', 'Audi_q5', 'Wuling_Hongguang', 'Tesla_modely', 'Peugeot_4008', 'Volvo_S60', 'ToyodaYARISL', 'Jianghuai_RuiWind', 'Baojun_560', 'Skoda_HaoRui', 'Hyundai_Tucson', 'Citroen_Picasso', 'BMW_7Series', 'Byd_Qin', 'Buick_Veran', 'Nissan_Loulan', 'Peugeot_307', 'Roewe_rx5', 'Hyundai_Rena', 'Faw_PentiumB30', 'TheGreatWall_C30', 'Lexus_RX', 'Toyota_Leiling', 'Volvo_V40', 'Ford_Escape', 'Modern_NameMap', 'EastWind_E70', 'Toyota_Vios', 'Chevrolet_Coruse', 'Nissan_Eida', 'Byd_Yuan', 'Volkswagen_Touareg', 'Ford-evos', 'BuickEnvision', 'Volvo_XC40', 'Ssangyong_EnjoyImperial', 'Honda_Lampai', 'Toyota_Crown', 'Kia_Gaale', 'Nissan_LIXil', 'Audi_a4l', 'Nissan_BluebirdClassic', 'LandRover_DiscoverySportEdition', 'Jianghuai_RuifengM4', 'Gac_TrumpCHIM8', 'Roewe_IMAX8', 'Nissan_Bluebird', 'Buick_GL8', 'Byd_Seagull', 'Benz_CClass', 'Mitsubishi_Oland', 'Toyota_Linfang', 'Cheetah_BlackGoldGang', 'Toyota_LandcoolLuze', 'Haver_h6coupe', 'Haver_H9', 'EastWind_JingyiX3', 'BMW3Series', 'AudiA3', 'Zhonghua_JunjieFRV', 'Audi_a6l', 'Faw_PentiumX80', 'TheGreatWall_M4', 'Modern_Leading', 'Benz_GLA', 'Kia_Run', 'Chevrolet_Lefeng', 'Ideal_L8', 'Changan_YueXiang', 'Modern_RuiYi', 'Changan_YuexiangV3', 'Peugeot_301', 'Buick_Excelle', 'Volkswagen_Touran', 'WeiBrand_VV7', 'Ford_Explorer', 'Jianghuai_RuifengM3', 'Gac_TrumpCHIGA8', 'Audi_rs7', 'Chery_Irizer5', 'Haver_h5', 'Baojun_310', 'Public_MusicExploration', 'Faw_Weizhi', 'BMW_1Series', 'KiaK3S', 'Seahorse_FortuneStar', 'Honda_FengFan', 'Public_Access', 'Changan_cx20', 'Byd_e3', 'Honda_Jade', 'PentiumB50', 'LandRover_RangeRover', 'Honda_InshiPie', 'Jeep_Guide', 'LandRover_DiscoveryWalk', 'Mazda3Exela', 'Gac_TrumpCHIGS8', 'Skoda_Cormick', 'Skoda_FastPie', 'Hafei_HorseRacing', 'Geely_BinYue', 'Skoda_Octavia', 'KiaK2', 'Changan_uni-v', 'Changan_CS35', 'Ideal_ONE', 'Changan_Gorgeous', 'China_JunjieFSV', 'Toyota_Camry', 'Honda_XR-V', 'Baojun_730', 'Modern_ix25', 'Roewe_rx5max', 'Chery_Tigo5X', 'BMW_X1', 'EastWind_JingyiX5', 'Kia_K5', 'Lexus_NX', 'Chevrolet_Chuangku', 'BYD_TangDM', 'VolkswagenID.4CROZZ', 'Faw_PentiumB50', 'Mazda_5', 'Toyota_Prado', 'Volkswagen_TiguanL', 'Geely_EmgrandL', 'Volkswagen_POLO', 'Audi_A5', 'Benz_CLS', 'Honda_Fit', 'EastWind_Jingyi', 'BuickEnclave.', 'Chery_A5', 'Lynk_01', 'GeelyICON', 'LandRover_RangeRoverSport', 'Skoda_Sharp', 'Kia_SmartRun', 'Geely_VisionX3', 'Haval_BigDog', 'Public_Tanko', 'Volkswagen_Tuyue', 'Lynk_02', 'Geely_Vision', 'Chery_Tigo3', 'Benz_GLC', 'Toyota_Reiz', 'Zhonghua_H530', 'Volkswagen_JettaVS7', 'Subaru_Forester', 'Geely_GX7', 'Nissan_Paladin', 'Volkswagen_TuyueX', 'Nissan_Tule', 'Chery_Tigo5', 'Geely_XingyueL', 'Cherie_Arezer', 'Changan_CS55PLUS', 'Toyota_Yizhi', 'Volkswagen_Passat', 'Link_05', 'Hyundai_YueNa', 'Changan_Escape', 'Volkswagen_JettaVA3', 'Toyota_YARiSL', 'Public_Driving', 'EastWind_PopularT5L', 'Mitsubishi_JinHyunASX', 'Lexus_CT', 'Ford_SharpWorld', 'Volkswagen_Beetle', 'Haver_h2s', 'Chery_Fengyun', 'Changan_CS35PLUS', 'Haver_h2', 'Volkswagen_Sharan', 'Volvo_XC60', 'Toyota_ViosFS', 'Chevy_Copaci.', 'LincolnMKZ', 'Baojun_630', 'Subaru_Outback', 'Changan_CS15', 'Link_03', 'Modern_Sonata', 'Beijing_BJ40', 'Wuling_Xingchi', 'Public_FilmExploration', 'Peugeot_5008', 'ZotYE_SR9', 'Citroen_C5', 'Chevy_Cowarz', 'BenzGLE', 'Peugeot_3008', 'Celis_AsktheWorldM5', 'Mercedes-benz_S-Class', 'Zhonghua_V5', 'HondaUR-V', 'Honda_Sidi', 'Benz_GLK', 'Byd_S7', 'Faw_Xiali', 'Roewe_rx8', 'Geely_EmgrandGS', 'Jianghuai_RuifengM5', 'Volkswagen_Phaeton', 'Volkswagen_Tiguan', 'Audi_q7', 'Zhonghua_V3', 'AudiQ3', 'MAXUSG50', 'Chery_Arezer3', 'Kia_Kessen', 'WeiBrand_Tank300', 'ChanganAuchan_X7PLUS', 'Geely_Xingyue', 'Jeep_Wrangler', 'Baic_ViwangM20', 'Volvo_XCClassic', 'Byd_s6', 'Cadillac_ats-l', 'MazDA_CX4', 'EastWind_PopularSX6', 'WeiBrand_VV6', 'Mitsubishi_Feiteng', 'Toyota_Izawa', 'Honda_Elisen', 'Buick_GL6', 'Benz_BClass', 'Suzuki_Vitra', 'Toyota_Fortuner', 'KaiChen_BigVDD-i', 'Suzuki_BigDipper', 'Hongq_hs5', 'Volkswagen_CC', 'Lexus_GX', 'Honda_HaoYing', 'Cheeta_q6', 'CadillacXT4', 'Cheeta_cs10', 'Buick_Regal', 'Haval_DivineBeast', 'Audi_q5l', 'Changan_WingStroke', 'BMW_X5', 'Citroen_Sega', 'Jetway_x70m', 'FAW_PentiumT77', 'Faw_PentiumB70', 'PentiumX80', 'EastWind_Scenery330', 'Honda_EnjoyArea', 'MINI_Cooper', 'LandRover_RangeRoverEvoque', 'WeiBrand_VV5', 'EastWind_WindGodAX7', 'Public_TanyueGTE', 'Byd_QinNewEnergy', 'Nissan_Energizer', 'Audi_tt', 'Buick_MicroBlue6', 'Chevy_Adventurer', 'TheGreatWall_C50', 'Baojun_RS-5', 'Cadillac_CT6', 'Ford_Collar', 'Baojun_307', 'Wuling_Glory', 'LandRover_Discovery', 'Toyota_Yaris', 'Byd_SongMAX', 'Toyota_Veranda', 'Chery_FlagCloud2', 'Ford_Win', 'GAC_TrumpCHIM6', 'Peugeot_208', 'Toyota_C-HR', 'Byd_Suirui', 'Kia_Rio', 'KiaSoul', 'Dongfeng_JingyiS50', 'Mazda_8', 'Suzuki_TianyuSX4', 'Roewe_RX3', 'Volkswagen_Langxing', 'Usheng_S330', 'Roewe_550', 'Nio_ES6', 'BMW_2Series', 'Changan_AuchanZ6', 'Geely_VisionX6', 'Kia_Sorento', 'Baojun_530', 'Cadillac_SRX', 'ZoTYE_T700', 'Mercedes-benzE-Class', 'Volvo_v60', 'Ford_Ruiji', 'Renault_Koreo', 'Lifan_XuanLang', 'Baojun_rc6', 'Peugeot_508', 'Modern_Paristi', 'Gac_TrumpCHIGS3', 'Haver_h7', 'TheGreatWall_M2', 'Volvo_S90', 'Gac_TrumpCHIGS5', 'Citroen_C4', 'Infiniti_QX50', 'BMW_X7', 'Beijing_X3', 'Dodge_Coolway', 'Volkswagen_C-TREK', 'Geely_EmgrandLHiP', 'Mitsubishi_WingedGod', 'Beijing_BJ80', 'MINI_ONE', 'Toyota_Sena', 'Byd_Destroyer', 'Changan_uni-t', 'Byd_HanDM', 'BYD_SongPLUSEV', 'Toyota_CrownLandRelease', 'Wuling_Stars', 'Chevrolet_Malibu', 'Chery_Tigo8', 'Haver_f7', 'Toyota_Senna', 'Modern_Nameplate', 'Chery_RiichM1', 'Byd_QinPLUSNewEnergy', 'Chery_Tigress', 'BMW_X6', 'EastWind_Scenery580', 'Modern_i30', 'GreatWall_Gun', 'Mitsubishi_Unknown', 'Benz_AClass', 'Hafei_PublicOpinion', 'Zhidou_D1', 'Baic_Viwang306', 'Changan_YuexiangV7', 'Baojun_150', 'Benz_Vito', 'Chevy_View', 'Lantu_Dreamers', 'EastWind_PopularS500', 'Faw_SenyaM80', 'Denza_D9', 'Ford_Lingyu', 'Lincoln_TheAviator', 'MaxusV80', 'DoubleRings_Littlearistocrats', 'Modern_SantaFe', 'Byd_G3', 'Jiangling_E200N', 'Haima_FumeiLaiF5', 'Link_09', 'NIO_ES8', 'Porsche-cayenne', 'Baojun_KIWi', 'Mazda_2Zest', 'Peugeot_2008', 'Volvo_S40', 'Hongq_e-qm5', 'CheryA3', 'Ford-forus', 'Toyota_CorollaSharpening', 'Toyota_Privia', 'Changan_Ouliwei', 'Buick_Encore', 'Modern_Veloster', 'Hippocampus_Cupid', 'Benz_CLA', 'Faw_Senya', 'BYD_Dolphin', 'Haver_h1', 'Geely_Haoyue', 'Mitsubishi_Pajero', 'Audi_q8', 'Baic_MagicSpeedS3', 'Byd_QinDM', 'Haval_ThunderMAX', 'Qoros_3', 'Lexus_EX', 'Peugeot_308', 'Volkswagen_TouranL', 'Fiat500', 'FAW_WeizhiV2', 'Ford_Wing-Beat', 'KiaKX5', 'Benz_GLS', 'Aud_q5Sportback', 'Toyota_Gravia', 'EastWind_DemeanorMX6', 'EastWind_Scenery', 'Qoros5', 'GoldenCup_SeaLion', 'DongfengKaiChen_KaiChenD60', 'Wuling_HongguangS3', 'Baojun_360', 'GoldCup_SmartS30', 'Mitsubishi_YiSong', 'Porsche_macan', 'Zhonghua_H3', 'Chery_E5', 'MazDA_CX-30', 'Baojun_rc5', 'Skoda_Cordiac', 'Lincoln_MKC', 'Public_Tukai', 'Jietu_X70', 'PengP5', 'Southeast_A5WingDance', 'Toyota_Lingshang', 'Roewe_I5', 'Volkswagen-t-rocSongExploration', 'GoldenCup_SeaLionKing', 'Modern_Feisi', 'Changan_BenbenMINI', 'Modern_ShengDAClassic', 'LandWind_LandWindX7', 'Opal_Vida', 'BenzClassV', 'Ideal_L9', 'Changan_uni-k', 'Audi_a8l', "Public_Hui'an", 'Chevy_Trailblazer', 'Skoda_Jingrui', 'Skoda_Koroc', 'Idea_S1', 'Mercedes-Benz_MaybachS-Class', 'Audi_s4', 'Toyota_CR-V', 'Ford_Fiesta', 'Volkswagen_Lamdo', 'Suzuki_Fronto', 'SeaLion_X30', 'Ford_TheRoadShaker', 'Volkswagen_Kaidi', 'MG_ZS', 'Toyota_Prius', 'Zhonghua_Junjie', 'Mazda_2', 'GoldCup_Kreis', 'Zhonghua_Zunchi', 'Suzuki_Alto', 'Geely_StarRui', 'smart_forfour', 'Citroen_Triumph', 'Chery_Tigo7', 'Zhonghua_H230', 'Flag_ds7', 'Chevrolet_MusicRV', 'Honda_Siming', 'Chevrolet_MusicStyle', 'Audi_a8', 'Changan_AuchanA600', 'TheGreatWall_C20R', 'Changan_AuchanX70A', 'Modern_Equus', 'Lincoln_Adventurers', 'Infiniti_QX30', 'Maserati_Levante', 'Lincoln_Navigator', 'Faw_XiALIN7', 'Borgward_BX5', 'Cadillac_ct5', 'Kia_Cerato', 'LandRover_Godwalker', 'Fukuda_SceneryG5', 'Jianghuai_RuifengS5', 'Mazda_Raywing', 'Geely_VisionX1', 'Chery_FlagCloud', 'Mazda_3StarCheng', 'Faw_PentiumB', 'Changan_AuchanX5', 'Ford-leadrui', 'Mitsubishi_PowerDazzle', 'Nissan_Tuca', 'Toyota_Overbearing', 'Chery_E3', 'Infiniti_GSeries', 'FAW_PentiumB90', 'BYD_Tang', 'Mazda_cx8', 'LandRover_Godwalker2', 'CadillacATSL', 'Chevrolet_Lezzi', 'FAW_ORANG', 'Jianghuai_Harmony', 'Dodge_Kubo', 'Peng_P7', 'Porsche-panamera', 'KaiChen_BigV', 'Huaqi_300E', 'Honda_Gori', 'BYD_F0', 'Jietu_X70PLUS')
    # CLASSES = ('Tissue_Paper Thin', 'Toiletries_Saky Toothpaste', 'Beverage_Nongfu Spring Jasmine Tea', 'Toiletries_Toothbrush', 'Alcohol_Snow Beer', 'Stationery_Glue Stick', 'Instant Drink_Huangwei Black Sesame Paste', "Snacks_Lay's Cucumber Flavor Potato Chips", 'Condiment_Monosodium Glutamate (MSG)', 'Condiment_China Salt', 'Instant Noodles_Pickled Mustard Beef Noodles', 'Dessert_Oreo Mini Cocoa Pastry', 'Toiletries_Liushen Soap', 'Puffed Food_Onion Chips (Blue)', 'Canned Food_Luncheon Meat', 'Beverage_Coffee', 'Condiment_Orange Tree Refined Sea Salt', 'Milk_Wangzai Milk', 'Toiletries_Safeguard Soap', 'Candy_Skittles Jar', 'Condiment_Xizaiji Oil and Vinegar Dressing', 'Condiment_Heinz Tomato Ketchup', 'Instant Noodles_Baixiang Golden Soup Fatty Beef', 'Canned Food_Wahaha Xylitol Porridge', 'Canned Food_Ribbonfish Canned', 'Instant Drink_Youlemei Wheat Flavor', 'Toiletries_Lux Soap', 'Instant Noodles_Uni-President Seafood Ramen', 'Dried Fruit_Plums', 'Chewing Gum_Plummed Tablets', 'Tissue_Tissue', 'Tissue_Boxed Tissue', 'Candy_Strawberry Candy', 'Condiment_Soy Sauce', "Chocolate_Hershey's Chocolate", 'Chocolate_Q Beans', 'Beverage_Six Walnuts', 'Toiletries_Hand Soap', 'Candy_Charcoal Roasted Coffee Candy', 'Gum_Five5 Watermelon Flavor Gum', 'Dried Fruit_Yitian Nuts', 'Dessert_Chocolate Wafer', 'Beverage_Nongfu Spring Mineral Water', 'Dried Fruit_Zhenglin Sunflower Seeds', 'Canned Food_Wahaha Longan and Lotus Seed Porridge', 'Dried Fruit_Liuliumei Green', 'Dried Snacks_Laojuechu Spicy Peanuts', "Snack_Food_Lay's Potato Chips (Bagged)", 'Puffed Food_Want Want Mini Crisps', 'Canned Food_Xiduoduo Goji Berry Porridge', 'Candy_Want Want QQ Gummies', 'Candy_Want Want Milk Candy', 'Dried Fruit_Crystal Lemon Slices', 'Dried Fruit_Hawthorn Blocks', 'Dried Fruit_Liuliumei Powder', 'Milk_Soy Milk', 'Milk_Anmuxi', 'Instant Noodles_Haidilao Tomato Beef Soup Noodles', 'Snacks_Haoyouqu Potato Chips Bagged', 'Candy_Animal Gummies', 'Seasoning_Li Salt', 'Milk_Coffee Milk', 'Chewing Gum_Doublemint Lozenges', 'Dessert_Chip Delight Soft Cookies', 'Condiment_Qishen Horseradish', 'Chocolate_Kinder Chocolate', 'Beverage_Red Bull', 'Candy_Skittles Bag', 'Stationery_Scissors', 'Dessert_Quduo Mini Cookies', 'Instant Drink_Nescafé Absolute Deep Black', 'Canned_Fish Roe Caviar', "Puffed Snacks_Lay's Potato Chips Canister", 'Stationery_Gel Pen', 'Snacks_Ya Potato', 'Dried Fruit_Qiaqia Hi Sunflower Seeds', 'Milk_Mengniu Boxed', 'Dried Fruit_Strange Flavor Beans', 'Instant Noodles_Lucky Bear Spicy Sour Powder', 'Alcohol_Tiger Beer', 'Dried Fruit_Dried Tangerine Peel', 'Stationery_Notebook', 'Chocolate_JiuJiu Dark Chocolate', 'Personal Care_Shanghai Medicinal Soap', 'Dried Fruit_Green Peas', 'Candy_Popping Candy', 'Dried Fruits_Qiaqia Caramel Melon Seeds', 'Chocolate_Mylikes', 'Milk_Wangzai Milk Can', 'Dried Fruit_Castanets', 'Stationery_Eraser', 'Instant Noodles_Spicy Beef Noodles', 'Chocolate_Mylikes Coconut Flavor', 'Dessert_Oreo Chocolate Flavor', 'Instant Drink_Yo-lo-mei', 'Chewing Gum_Mentos', 'Dried Fruit_Yangmei', 'Snacks_Friend Fun Steak Flavored Chips', 'Condiment_Qishen Black Pepper Powder', 'Alcohol_Snow Beer Blue', 'Candy_Orion Orange Candy', 'Instant Noodles_Spicy Beef Noodles', 'Instant Noodles_Tomato Egg Beef Noodles', 'Instant Noodles_Shrimp Fish Cake Noodles', 'Dessert_Chocolate Pie', 'Drink_Coconut Tree Coconut Juice', 'Drink_Sprite', 'Drink_Coca-Cola', 'Drink_Suntory Oolong Tea', 'Drink_Enen Soda Water', 'Dessert_Oreo Original Flavor', 'Canned_Fresh Stewed Tremella by Xiduoduo', 'Dessert_Q Ti', 'Chocolate_Green Chocolate', 'Tissue_Banbu Facial Tissue', "Snacks_Lay's Barbecue Flavored Chips", 'Milk_AD Calcium Milk', 'Beverage_Dalian Soda', 'Personal Care_Tissues', 'Stationery_Correction Tape', 'Dry Fruits_Mustard Peanuts')

    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
               (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
               (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
               (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
               (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
               (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
               (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
               (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
               (134, 134, 103), (145, 148, 174), (255, 208, 186),
               (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
               (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
               (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
               (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
               (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
               (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
               (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
               (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
               (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
               (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
               (191, 162, 208)]

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].rsplit('.', 1)[0] + self.seg_suffix

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def _proposal2json(self, results):
        """Convert proposal results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results

    def _segm2json(self, results):
        """Convert instance segmentation results to COCO json style."""
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        gt_bboxes = []
        for i in range(len(self.img_ids)):
            ann_ids = self.coco.get_ann_ids(img_ids=self.img_ids[i])
            ann_info = self.coco.load_anns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir

    def evaluate_det_segm(self,
                          results,
                          result_files,
                          coco_gt,
                          metrics,
                          logger=None,
                          classwise=False,
                          proposal_nums=(100, 300, 1000),
                          iou_thrs=None,
                          metric_items=None):
        """Instance segmentation and object detection evaluation in COCO
        protocol.

        Args:
            results (list[list | tuple | dict]): Testing results of the
                dataset.
            result_files (dict[str, str]): a dict contains json file path.
            coco_gt (COCO): COCO API object with ground truth annotation.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        eval_results = OrderedDict()
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                if isinstance(results[0], tuple):
                    raise KeyError('proposal_fast is not supported for '
                                   'instance segmentation result.')
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = mmcv.load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                    warnings.simplefilter('once')
                    warnings.warn(
                        'The key "bbox" is deleted for more accurate mask AP '
                        'of small/medium/large instances since v2.12.0. This '
                        'does not change the overall mAP calculation.',
                        UserWarning)
                coco_det = coco_gt.loadRes(predictions)
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            cocoEval = COCOeval(coco_gt, coco_det, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.evaluate()
                cocoEval.accumulate()

                # Save coco summarize print information to logger
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()
                print_log('\n' + redirect_string.getvalue(), logger=logger)

                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate()

                # Save coco summarize print information to logger
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()
                print_log('\n' + redirect_string.getvalue(), logger=logger)

                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                    )
                    eval_results[key] = val
                ap = cocoEval.stats[:6]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    f'{ap[4]:.3f} {ap[5]:.3f}')

        return eval_results

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        coco_gt = self.coco
        self.cat_ids = coco_gt.get_cat_ids(cat_names=self.CLASSES)

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        eval_results = self.evaluate_det_segm(results, result_files, coco_gt,
                                              metrics, logger, classwise,
                                              proposal_nums, iou_thrs,
                                              metric_items)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
