# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import List, Union

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .api_wrappers import COCO
from .base_det_dataset import BaseDetDataset


@DATASETS.register_module()
class CocoDataset(BaseDetDataset):
    """Dataset for COCO."""

    METAINFO = {
        'classes':
        # ('scooter_Hero_Maestro','car_TataMotors_Zest','motorcycle_Hero_Passion','car_TataMotors_Indica','car_Chevrolet_Spark','car_Nissan_Terrano','scooter_Suzuki_Swish','motorcycle_Honda_KarizmaZMR','autorickshaw_Piaggio','car_MarutiSuzuki_Swift','car_Nissan_Micra','motorcycle_TVS_StarCityPlus','truck_BharatBenz','car_Hyundai_Eon','motorcycle_RoyalEnfield_Thunderbird350','car_Honda_Civic','car_Hyundai_Xcent','car_Renault_Duster','car_TataMotors_Tigor','car_Mercedes-Benz_E-Class','motorcycle_TVS_ExcelHeavyDuty','car_TataMotors_Hexa','scooter_Vespa_VXL125','car_Mercedes-Benz_A-Class','car_Honda_Amaze','car_MarutiSuzuki_Celerio','scooter_Honda_Activa','truck_Eicher','scooter_TVS_Wego','car_MarutiSuzuki_Ciaz','car_Honda_Accord','car_Chevrolet_Tavera','motorcycle_Hero_PassionPlus','car_Ford_Ikon','motorcycle_Yamaha_Fazer','motorcycle_Bajaj_PulsarRS200','car_Jeep_Compass','car_Bmw_X1','car_MarutiSuzuki_Dzire','motorcycle_Bajaj_Pulsar200','motorcycle_Mahindra_Centuro','motorcycle_RoyalEnfield_Classic350','motorcycle_Yamaha_R15','scooter_Honda_Dio','car_Mahindra_Scorpio','car_Audi_A3','car_MarutiSuzuki_WagonR','car_MarutiSuzuki_Baleno','motorcycle_Hero_HFDeluxe','motorcycle_TVS_ApacheRTR200','car_Jeep_Wrangler','motorcycle_Bajaj_Discover','autorickshaw_Others','car_Volvo_Xc60','truck_AshokLeyland','motorcycle_Yamaha_FZS-FI','car_Skoda_Rapid','scooter_TVS_Jupiter','car_Chevrolet_Enjoy','motorcycle_Others','car_Skoda_Superb','motorcycle_HeroHonda_SplendorNXG','motorcycle_Bajaj_PulsarNS200','motorcycle_TVS_ApacheRTR160','motorcycle_Honda_Karizma','motorcycle_Hero_Hunk','motorcycle_KTM_Duke','scooter_Suzuki_Burgman','car_Toyota_Fortuner','motorcycle_Covered','car_MarutiSuzuki_Ertiga','car_Mahindra_Bolero','motorcycle_RoyalEnfield_Thunderbird350X','car_Mahindra_Xylo','car_Chevrolet_Cruze','motorcycle_Yamaha_RX100','motorcycle_Honda_CBTwister','car_Volkswagen_Polo','motorcycle_Bajaj_Discover100','car_MarutiSuzuki_Zen','car_Mercedes-Benz_S-Class','motorcycle_Bajaj_Discover125','car_Hyundai_I20','scooter_Honda_Grazia','car_Honda_Wr-V','motorcycle_TVS_Sport','car_Volkswagen_Vento','car_MarutiSuzuki_SX4','scooter_Hero_Duet','scooter_TVS_Streak','scooter_Hero_Pleasure','car_Ford_Fiesta','car_Honda_Brio','autorickshaw_Bajaj','car_Mercedes-Benz_C-Class','scooter_Mahindra_Gusto','scooter_Suzuki_Access','motorcycle_TVS_XL100','motorcycle_Honda_StunnerCBF','car_MarutiSuzuki_Esteem2000','motorcycle_Hero_PassionPro','car_MarutiSuzuki_S-Cross','car_TataMotors_Nano','motorcycle_Hero_Glamour','motorcycle_Suzuki_Samurai','car_Bmw_2-Series-220d','car_Volkswagen_Ameo','car_MarutiSuzuki_AltoK10','car_Skoda_Fabia','car_Toyota_Corolla','motorcycle_Hero_XPulse200','motorcycle_RoyalEnfield_Bullet350','car_Chevrolet_Beat','car_Renault_Lodgy','car_Hyundai_I10','car_Audi_Q3','car_Mahindra_XUV500','car_Honda_Brv','car_Bmw_3-Series','motorcycle_Suzuki_Slingshot','car_Chevrolet_Aveo','motorcycle_RoyalEnfield_Interceptor650','car_Others','car_Mahindra_Reva','motorcycle_Bajaj_Avenger','car_Toyota_Qualis','motorcycle_Bajaj_Discover135','autorickshaw_TVS','car_Hindustan_Ambassador','motorcycle_Suzuki_Gixxer','car_Ford_Ecosport','car_Hyundai_Santro','car_MarutiSuzuki_Ignis','motorcycle_RoyalEnfield_Classic500','motorcycle_RoyalEnfield_Meteor350','car_MarutiSuzuki_VitaraBrezza','car_Fiat_PuntoEvo','car_Toyota_EtiosLiva','motorcycle_Yamaha_FZ25','motorcycle_Bajaj_Platina','motorcycle_Yamaha_Crux','car_Ford_Everest','motorcycle_Honda_Shine','scooter_Others','car_Ford_Figo','motorcycle_Bajaj_CT100','car_Mercedes-Benz_Gla-Class','car_Mercedes-Benz_AmgGt4-DoorCoupe','scooter_TVS_Zest','autorickshaw_Mahindra','car_Fiat_Linea','motorcycle_Bajaj_Pulsar220F','car_MarutiSuzuki_Omni','car_Renault_Kwid','car_Honda_Cr-V','motorcycle_Bajaj_V15','scooter_Bajaj_Chetak','car_MarutiSuzuki_Ritz','car_Mahindra_Verito','motorcycle_RoyalEnfield_ContinentalGT650','motorcycle_Bajaj_Discover110','bus','motorcycle_RoyalEnfield_Bullet500','truck_Others','car_Renault_Scala','car_Toyota_Innova','car_Nissan_Sunny','car_Toyota_Etios','car_Mahindra_TUV300','motorcycle_Hero_SuperSplendor','car_Covered','car_MarutiSuzuki_1000','car_TataMotors_Nexon','car_Hyundai_Creta','mini-bus_Others','motorcycle_TVS_Excel100','motorcycle_Bajaj_V12','car_TataMotors_Sumo','car_Honda_Jazz','car_Hyundai_Accent','scooter_Yamaha_RayZR','car_Volkswagen_Jetta','autorickshaw_Covered','motorcycle_Yamaha_Libero','car_Skoda_Octavia','car_MarutiSuzuki_Alto800','scooter_Yamaha_Fascino125','autorickshaw_Atul','motorcycle_Honda_Unicorn','car_Force_TraxToofan','car_Mercedes-Benz_G-Class','car_Ford_EcoSportTitanium','truck_Mahindra','scooter_TVS_Pep','car_TataMotors_Indigo','car_MarutiSuzuki_Eeco','scooter_TVS_Ntorq','truck_Tata','truck_SML','car_Volvo_Xc40','motorcycle_Bajaj_Pulsar180','car_Renault_Logan','motorcycle_HeroHonda_CBZ','car_TataMotors_Tiago','motorcycle_Honda_CBHornet160R','car_Hyundai_Verna','car_Ford_Aspire','motorcycle_Honda_SP125','motorcycle_Hero_Splendor','motorcycle_Yamaha_FZ-V3','car_TataMotors_Safari','scooter_Honda_Aviator','car_Honda_City','motorcycle_Bajaj_Pulsar150','motorcycle_TVS_Victor'),

        # ('1_puffed_food', '2_puffed_food', '3_puffed_food', '4_puffed_food', '5_puffed_food', '6_puffed_food', '7_puffed_food', '8_puffed_food', '9_puffed_food', '10_puffed_food', '11_puffed_food', '12_puffed_food', '13_dried_fruit', '14_dried_fruit', '15_dried_fruit', '16_dried_fruit', '17_dried_fruit', '18_dried_fruit', '19_dried_fruit', '20_dried_fruit', '21_dried_fruit', '22_dried_food', '23_dried_food', '24_dried_food', '25_dried_food', '26_dried_food', '27_dried_food', '28_dried_food', '29_dried_food', '30_dried_food', '31_instant_drink', '32_instant_drink', '33_instant_drink', '34_instant_drink', '35_instant_drink', '36_instant_drink', '37_instant_drink', '38_instant_drink', '39_instant_drink', '40_instant_drink', '41_instant_drink', '42_instant_noodles', '43_instant_noodles', '44_instant_noodles', '45_instant_noodles', '46_instant_noodles', '47_instant_noodles', '48_instant_noodles', '49_instant_noodles', '50_instant_noodles', '51_instant_noodles', '52_instant_noodles', '53_instant_noodles', '54_dessert', '55_dessert', '56_dessert', '57_dessert', '58_dessert', '59_dessert', '60_dessert', '61_dessert', '62_dessert', '63_dessert', '64_dessert', '65_dessert', '66_dessert', '67_dessert', '68_dessert', '69_dessert', '70_dessert', '71_drink', '72_drink', '73_drink', '74_drink', '75_drink', '76_drink', '77_drink', '78_drink', '79_alcohol', '80_alcohol', '81_drink', '82_drink', '83_drink', '84_drink', '85_drink', '86_drink', '87_drink', '88_alcohol', '89_alcohol', '90_alcohol', '91_alcohol', '92_alcohol', '93_alcohol', '94_alcohol', '95_alcohol', '96_alcohol', '97_milk', '98_milk', '99_milk', '100_milk', '101_milk', '102_milk', '103_milk', '104_milk', '105_milk', '106_milk', '107_milk', '108_canned_food', '109_canned_food', '110_canned_food', '111_canned_food', '112_canned_food', '113_canned_food', '114_canned_food', '115_canned_food', '116_canned_food', '117_canned_food', '118_canned_food', '119_canned_food', '120_canned_food', '121_canned_food', '122_chocolate', '123_chocolate', '124_chocolate', '125_chocolate', '126_chocolate', '127_chocolate', '128_chocolate', '129_chocolate', '130_chocolate', '131_chocolate', '132_chocolate', '133_chocolate', '134_gum', '135_gum', '136_gum', '137_gum', '138_gum', '139_gum', '140_gum', '141_gum', '142_candy', '143_candy', '144_candy', '145_candy', '146_candy', '147_candy', '148_candy', '149_candy', '150_candy', '151_candy', '152_seasoner', '153_seasoner', '154_seasoner', '155_seasoner', '156_seasoner', '157_seasoner', '158_seasoner', '159_seasoner', '160_seasoner', '161_seasoner', '162_seasoner', '163_seasoner', '164_personal_hygiene', '165_personal_hygiene', '166_personal_hygiene', '167_personal_hygiene', '168_personal_hygiene', '169_personal_hygiene', '170_personal_hygiene', '171_personal_hygiene', '172_personal_hygiene', '173_personal_hygiene', '174_tissue', '175_tissue', '176_tissue', '177_tissue', '178_tissue', '179_tissue', '180_tissue', '181_tissue', '182_tissue', '183_tissue', '184_tissue', '185_tissue', '186_tissue', '187_tissue', '188_tissue', '189_tissue', '190_tissue', '191_tissue', '192_tissue', '193_tissue', '194_stationery', '195_stationery', '196_stationery', '197_stationery', '198_stationery', '199_stationery', '200_stationery'),

        ('Fiat_Feixiang', 'Ford_Focus', 'Volkswagen_Magotan', 'Peugeot_408', 'Suzuki_KaiYue', 'Jeep_FreedomMan', 'Toyota_Corolla', 'Gac_TrumpCHIGS4', 'Baic_ViwangM30', 'Kia_Furedi', 'Volkswagen_lavida', 'Zotye_DamaiX7', 'Honda_Accord', 'BMW_5Series', 'Byd_F3', 'Nissan_Teana', 'Cadillac_cts', 'Changan_CS75', 'Changan_CS75PLUS', 'Honda_CrownRoad', 'Haver_m6', 'Hyundai_Shengda', 'AudiA4', 'Volkswagen_Jetta', 'Toyota_Elfa', 'Modern_HappyMovement', 'Toyota_RAV4Rongfang', 'Kia_Sportage', 'Mazda_6', 'other', 'Audi_a6', 'Chery_QQ', 'Jiangling_Yuhu5', 'Toyota_HILUX', 'unknown', 'Ford_Mondeo', 'Mazda_Atz', 'Mercedes-Benz_M-Class', 'Haver_h6', 'Honda_Civic', 'Lord6', 'Citroen_Elysee', 'Mazda_cx-5', 'Geely_EmgrandGL', 'Volkswagen_JettaVS5', 'Lucky_KingKong', 'Nissan_Sunshine', 'Nissan_Sylphy', 'Orient_Scenery580', 'Modern_Elantra', 'Benz_RClass', 'Nissan_Qijun', 'Citroen_C3-XR', 'Faw_XiALIN5', 'Volvo_XC90', 'Chery_Tigo3X', 'Toyota_Highlander', 'Mazda_3', 'Nissan_Qashqai', 'Byd_Song', 'Lexus_ux', 'Nissan_versa', 'BMW_X3', 'Lexus_ES', 'Volkswagen_Santana', 'Volkswagen_Golf', 'Hongq_h7', 'Roewe_EI5', 'Modern_ix35', 'Chevrolet_Sail', 'Chevrolet_Cruze', 'Volkswagen_Sagitar', 'Buick_Lacrosse', 'LandWind_X8', 'Jeep_FreeLight', 'Honda_Benzhi', 'Geely_BoYue', 'Benz_GLB', 'Honda_CR-V', 'Public_Wirin', 'Jianghuai_RuifengS3', 'Public_Bora', 'Chery_Tigo7Plus', 'Suzuki_Swift', 'Kia_K3', 'Geely_Emgrand', 'Roewe_350', 'Volkswagen_Tuang', 'RenaultCorrega', 'Dongfeng_Lingzhi', 'Baojun_510', 'Cherry_IrizerGX', 'Buick_Yinglong', 'Modern_Langdong', 'Honda_Odyssey', 'Audi_q5', 'Wuling_Hongguang', 'Tesla_modely', 'Peugeot_4008', 'Volvo_S60', 'ToyodaYARISL', 'Jianghuai_RuiWind', 'Baojun_560', 'Skoda_HaoRui', 'Hyundai_Tucson', 'Citroen_Picasso', 'BMW_7Series', 'Byd_Qin', 'Buick_Veran', 'Nissan_Loulan', 'Peugeot_307', 'Roewe_rx5', 'Hyundai_Rena', 'Faw_PentiumB30', 'TheGreatWall_C30', 'Lexus_RX', 'Toyota_Leiling', 'Volvo_V40', 'Ford_Escape', 'Modern_NameMap', 'EastWind_E70', 'Toyota_Vios', 'Chevrolet_Coruse', 'Nissan_Eida', 'Byd_Yuan', 'Volkswagen_Touareg', 'Ford-evos', 'BuickEnvision', 'Volvo_XC40', 'Ssangyong_EnjoyImperial', 'Honda_Lampai', 'Toyota_Crown', 'Kia_Gaale', 'Nissan_LIXil', 'Audi_a4l', 'Nissan_BluebirdClassic', 'LandRover_DiscoverySportEdition', 'Jianghuai_RuifengM4', 'Gac_TrumpCHIM8', 'Roewe_IMAX8', 'Nissan_Bluebird', 'Buick_GL8', 'Byd_Seagull', 'Benz_CClass', 'Mitsubishi_Oland', 'Toyota_Linfang', 'Cheetah_BlackGoldGang', 'Toyota_LandcoolLuze', 'Haver_h6coupe', 'Haver_H9', 'EastWind_JingyiX3', 'BMW3Series', 'AudiA3', 'Zhonghua_JunjieFRV', 'Audi_a6l', 'Faw_PentiumX80', 'TheGreatWall_M4', 'Modern_Leading', 'Benz_GLA', 'Kia_Run', 'Chevrolet_Lefeng', 'Ideal_L8', 'Changan_YueXiang', 'Modern_RuiYi', 'Changan_YuexiangV3', 'Peugeot_301', 'Buick_Excelle', 'Volkswagen_Touran', 'WeiBrand_VV7', 'Ford_Explorer', 'Jianghuai_RuifengM3', 'Gac_TrumpCHIGA8', 'Audi_rs7', 'Chery_Irizer5', 'Haver_h5', 'Baojun_310', 'Public_MusicExploration', 'Faw_Weizhi', 'BMW_1Series', 'KiaK3S', 'Seahorse_FortuneStar', 'Honda_FengFan', 'Public_Access', 'Changan_cx20', 'Byd_e3', 'Honda_Jade', 'PentiumB50', 'LandRover_RangeRover', 'Honda_InshiPie', 'Jeep_Guide', 'LandRover_DiscoveryWalk', 'Mazda3Exela', 'Gac_TrumpCHIGS8', 'Skoda_Cormick', 'Skoda_FastPie', 'Hafei_HorseRacing', 'Geely_BinYue', 'Skoda_Octavia', 'KiaK2', 'Changan_uni-v', 'Changan_CS35', 'Ideal_ONE', 'Changan_Gorgeous', 'China_JunjieFSV', 'Toyota_Camry', 'Honda_XR-V', 'Baojun_730', 'Modern_ix25', 'Roewe_rx5max', 'Chery_Tigo5X', 'BMW_X1', 'EastWind_JingyiX5', 'Kia_K5', 'Lexus_NX', 'Chevrolet_Chuangku', 'BYD_TangDM', 'VolkswagenID.4CROZZ', 'Faw_PentiumB50', 'Mazda_5', 'Toyota_Prado', 'Volkswagen_TiguanL', 'Geely_EmgrandL', 'Volkswagen_POLO', 'Audi_A5', 'Benz_CLS', 'Honda_Fit', 'EastWind_Jingyi', 'BuickEnclave.', 'Chery_A5', 'Lynk_01', 'GeelyICON', 'LandRover_RangeRoverSport', 'Skoda_Sharp', 'Kia_SmartRun', 'Geely_VisionX3', 'Haval_BigDog', 'Public_Tanko', 'Volkswagen_Tuyue', 'Lynk_02', 'Geely_Vision', 'Chery_Tigo3', 'Benz_GLC', 'Toyota_Reiz', 'Zhonghua_H530', 'Volkswagen_JettaVS7', 'Subaru_Forester', 'Geely_GX7', 'Nissan_Paladin', 'Volkswagen_TuyueX', 'Nissan_Tule', 'Chery_Tigo5', 'Geely_XingyueL', 'Cherie_Arezer', 'Changan_CS55PLUS', 'Toyota_Yizhi', 'Volkswagen_Passat', 'Link_05', 'Hyundai_YueNa', 'Changan_Escape', 'Volkswagen_JettaVA3', 'Toyota_YARiSL', 'Public_Driving', 'EastWind_PopularT5L', 'Mitsubishi_JinHyunASX', 'Lexus_CT', 'Ford_SharpWorld', 'Volkswagen_Beetle', 'Haver_h2s', 'Chery_Fengyun', 'Changan_CS35PLUS', 'Haver_h2', 'Volkswagen_Sharan', 'Volvo_XC60', 'Toyota_ViosFS', 'Chevy_Copaci.', 'LincolnMKZ', 'Baojun_630', 'Subaru_Outback', 'Changan_CS15', 'Link_03', 'Modern_Sonata', 'Beijing_BJ40', 'Wuling_Xingchi', 'Public_FilmExploration', 'Peugeot_5008', 'ZotYE_SR9', 'Citroen_C5', 'Chevy_Cowarz', 'BenzGLE', 'Peugeot_3008', 'Celis_AsktheWorldM5', 'Mercedes-benz_S-Class', 'Zhonghua_V5', 'HondaUR-V', 'Honda_Sidi', 'Benz_GLK', 'Byd_S7', 'Faw_Xiali', 'Roewe_rx8', 'Geely_EmgrandGS', 'Jianghuai_RuifengM5', 'Volkswagen_Phaeton', 'Volkswagen_Tiguan', 'Audi_q7', 'Zhonghua_V3', 'AudiQ3', 'MAXUSG50', 'Chery_Arezer3', 'Kia_Kessen', 'WeiBrand_Tank300', 'ChanganAuchan_X7PLUS', 'Geely_Xingyue', 'Jeep_Wrangler', 'Baic_ViwangM20', 'Volvo_XCClassic', 'Byd_s6', 'Cadillac_ats-l', 'MazDA_CX4', 'EastWind_PopularSX6', 'WeiBrand_VV6', 'Mitsubishi_Feiteng', 'Toyota_Izawa', 'Honda_Elisen', 'Buick_GL6', 'Benz_BClass', 'Suzuki_Vitra', 'Toyota_Fortuner', 'KaiChen_BigVDD-i', 'Suzuki_BigDipper', 'Hongq_hs5', 'Volkswagen_CC', 'Lexus_GX', 'Honda_HaoYing', 'Cheeta_q6', 'CadillacXT4', 'Cheeta_cs10', 'Buick_Regal', 'Haval_DivineBeast', 'Audi_q5l', 'Changan_WingStroke', 'BMW_X5', 'Citroen_Sega', 'Jetway_x70m', 'FAW_PentiumT77', 'Faw_PentiumB70', 'PentiumX80', 'EastWind_Scenery330', 'Honda_EnjoyArea', 'MINI_Cooper', 'LandRover_RangeRoverEvoque', 'WeiBrand_VV5', 'EastWind_WindGodAX7', 'Public_TanyueGTE', 'Byd_QinNewEnergy', 'Nissan_Energizer', 'Audi_tt', 'Buick_MicroBlue6', 'Chevy_Adventurer', 'TheGreatWall_C50', 'Baojun_RS-5', 'Cadillac_CT6', 'Ford_Collar', 'Baojun_307', 'Wuling_Glory', 'LandRover_Discovery', 'Toyota_Yaris', 'Byd_SongMAX', 'Toyota_Veranda', 'Chery_FlagCloud2', 'Ford_Win', 'GAC_TrumpCHIM6', 'Peugeot_208', 'Toyota_C-HR', 'Byd_Suirui', 'Kia_Rio', 'KiaSoul', 'Dongfeng_JingyiS50', 'Mazda_8', 'Suzuki_TianyuSX4', 'Roewe_RX3', 'Volkswagen_Langxing', 'Usheng_S330', 'Roewe_550', 'Nio_ES6', 'BMW_2Series', 'Changan_AuchanZ6', 'Geely_VisionX6', 'Kia_Sorento', 'Baojun_530', 'Cadillac_SRX', 'ZoTYE_T700', 'Mercedes-benzE-Class', 'Volvo_v60', 'Ford_Ruiji', 'Renault_Koreo', 'Lifan_XuanLang', 'Baojun_rc6', 'Peugeot_508', 'Modern_Paristi', 'Gac_TrumpCHIGS3', 'Haver_h7', 'TheGreatWall_M2', 'Volvo_S90', 'Gac_TrumpCHIGS5', 'Citroen_C4', 'Infiniti_QX50', 'BMW_X7', 'Beijing_X3', 'Dodge_Coolway', 'Volkswagen_C-TREK', 'Geely_EmgrandLHiP', 'Mitsubishi_WingedGod', 'Beijing_BJ80', 'MINI_ONE', 'Toyota_Sena', 'Byd_Destroyer', 'Changan_uni-t', 'Byd_HanDM', 'BYD_SongPLUSEV', 'Toyota_CrownLandRelease', 'Wuling_Stars', 'Chevrolet_Malibu', 'Chery_Tigo8', 'Haver_f7', 'Toyota_Senna', 'Modern_Nameplate', 'Chery_RiichM1', 'Byd_QinPLUSNewEnergy', 'Chery_Tigress', 'BMW_X6', 'EastWind_Scenery580', 'Modern_i30', 'GreatWall_Gun', 'Mitsubishi_Unknown', 'Benz_AClass', 'Hafei_PublicOpinion', 'Zhidou_D1', 'Baic_Viwang306', 'Changan_YuexiangV7', 'Baojun_150', 'Benz_Vito', 'Chevy_View', 'Lantu_Dreamers', 'EastWind_PopularS500', 'Faw_SenyaM80', 'Denza_D9', 'Ford_Lingyu', 'Lincoln_TheAviator', 'MaxusV80', 'DoubleRings_Littlearistocrats', 'Modern_SantaFe', 'Byd_G3', 'Jiangling_E200N', 'Haima_FumeiLaiF5', 'Link_09', 'NIO_ES8', 'Porsche-cayenne', 'Baojun_KIWi', 'Mazda_2Zest', 'Peugeot_2008', 'Volvo_S40', 'Hongq_e-qm5', 'CheryA3', 'Ford-forus', 'Toyota_CorollaSharpening', 'Toyota_Privia', 'Changan_Ouliwei', 'Buick_Encore', 'Modern_Veloster', 'Hippocampus_Cupid', 'Benz_CLA', 'Faw_Senya', 'BYD_Dolphin', 'Haver_h1', 'Geely_Haoyue', 'Mitsubishi_Pajero', 'Audi_q8', 'Baic_MagicSpeedS3', 'Byd_QinDM', 'Haval_ThunderMAX', 'Qoros_3', 'Lexus_EX', 'Peugeot_308', 'Volkswagen_TouranL', 'Fiat500', 'FAW_WeizhiV2', 'Ford_Wing-Beat', 'KiaKX5', 'Benz_GLS', 'Aud_q5Sportback', 'Toyota_Gravia', 'EastWind_DemeanorMX6', 'EastWind_Scenery', 'Qoros5', 'GoldenCup_SeaLion', 'DongfengKaiChen_KaiChenD60', 'Wuling_HongguangS3', 'Baojun_360', 'GoldCup_SmartS30', 'Mitsubishi_YiSong', 'Porsche_macan', 'Zhonghua_H3', 'Chery_E5', 'MazDA_CX-30', 'Baojun_rc5', 'Skoda_Cordiac', 'Lincoln_MKC', 'Public_Tukai', 'Jietu_X70', 'PengP5', 'Southeast_A5WingDance', 'Toyota_Lingshang', 'Roewe_I5', 'Volkswagen-t-rocSongExploration', 'GoldenCup_SeaLionKing', 'Modern_Feisi', 'Changan_BenbenMINI', 'Modern_ShengDAClassic', 'LandWind_LandWindX7', 'Opal_Vida', 'BenzClassV', 'Ideal_L9', 'Changan_uni-k', 'Audi_a8l', "Public_Hui'an", 'Chevy_Trailblazer', 'Skoda_Jingrui', 'Skoda_Koroc', 'Idea_S1', 'Mercedes-Benz_MaybachS-Class', 'Audi_s4', 'Toyota_CR-V', 'Ford_Fiesta', 'Volkswagen_Lamdo', 'Suzuki_Fronto', 'SeaLion_X30', 'Ford_TheRoadShaker', 'Volkswagen_Kaidi', 'MG_ZS', 'Toyota_Prius', 'Zhonghua_Junjie', 'Mazda_2', 'GoldCup_Kreis', 'Zhonghua_Zunchi', 'Suzuki_Alto', 'Geely_StarRui', 'smart_forfour', 'Citroen_Triumph', 'Chery_Tigo7', 'Zhonghua_H230', 'Flag_ds7', 'Chevrolet_MusicRV', 'Honda_Siming', 'Chevrolet_MusicStyle', 'Audi_a8', 'Changan_AuchanA600', 'TheGreatWall_C20R', 'Changan_AuchanX70A', 'Modern_Equus', 'Lincoln_Adventurers', 'Infiniti_QX30', 'Maserati_Levante', 'Lincoln_Navigator', 'Faw_XiALIN7', 'Borgward_BX5', 'Cadillac_ct5', 'Kia_Cerato', 'LandRover_Godwalker', 'Fukuda_SceneryG5', 'Jianghuai_RuifengS5', 'Mazda_Raywing', 'Geely_VisionX1', 'Chery_FlagCloud', 'Mazda_3StarCheng', 'Faw_PentiumB', 'Changan_AuchanX5', 'Ford-leadrui', 'Mitsubishi_PowerDazzle', 'Nissan_Tuca', 'Toyota_Overbearing', 'Chery_E3', 'Infiniti_GSeries', 'FAW_PentiumB90', 'BYD_Tang', 'Mazda_cx8', 'LandRover_Godwalker2', 'CadillacATSL', 'Chevrolet_Lezzi', 'FAW_ORANG', 'Jianghuai_Harmony', 'Dodge_Kubo', 'Peng_P7', 'Porsche-panamera', 'KaiChen_BigV', 'Huaqi_300E', 'Honda_Gori', 'BYD_F0', 'Jietu_X70PLUS'),

        # ('Tissue_Paper Thin', 'Toiletries_Saky Toothpaste', 'Beverage_Nongfu Spring Jasmine Tea', 'Toiletries_Toothbrush', 'Alcohol_Snow Beer', 'Stationery_Glue Stick', 'Instant Drink_Huangwei Black Sesame Paste', "Snacks_Lay's Cucumber Flavor Potato Chips", 'Condiment_Monosodium Glutamate (MSG)', 'Condiment_China Salt', 'Instant Noodles_Pickled Mustard Beef Noodles', 'Dessert_Oreo Mini Cocoa Pastry', 'Toiletries_Liushen Soap', 'Puffed Food_Onion Chips (Blue)', 'Canned Food_Luncheon Meat', 'Beverage_Coffee', 'Condiment_Orange Tree Refined Sea Salt', 'Milk_Wangzai Milk', 'Toiletries_Safeguard Soap', 'Candy_Skittles Jar', 'Condiment_Xizaiji Oil and Vinegar Dressing', 'Condiment_Heinz Tomato Ketchup', 'Instant Noodles_Baixiang Golden Soup Fatty Beef', 'Canned Food_Wahaha Xylitol Porridge', 'Canned Food_Ribbonfish Canned', 'Instant Drink_Youlemei Wheat Flavor', 'Toiletries_Lux Soap', 'Instant Noodles_Uni-President Seafood Ramen', 'Dried Fruit_Plums', 'Chewing Gum_Plummed Tablets', 'Tissue_Tissue', 'Tissue_Boxed Tissue', 'Candy_Strawberry Candy', 'Condiment_Soy Sauce', "Chocolate_Hershey's Chocolate", 'Chocolate_Q Beans', 'Beverage_Six Walnuts', 'Toiletries_Hand Soap', 'Candy_Charcoal Roasted Coffee Candy', 'Gum_Five5 Watermelon Flavor Gum', 'Dried Fruit_Yitian Nuts', 'Dessert_Chocolate Wafer', 'Beverage_Nongfu Spring Mineral Water', 'Dried Fruit_Zhenglin Sunflower Seeds', 'Canned Food_Wahaha Longan and Lotus Seed Porridge', 'Dried Fruit_Liuliumei Green', 'Dried Snacks_Laojuechu Spicy Peanuts', "Snack_Food_Lay's Potato Chips (Bagged)", 'Puffed Food_Want Want Mini Crisps', 'Canned Food_Xiduoduo Goji Berry Porridge', 'Candy_Want Want QQ Gummies', 'Candy_Want Want Milk Candy', 'Dried Fruit_Crystal Lemon Slices', 'Dried Fruit_Hawthorn Blocks', 'Dried Fruit_Liuliumei Powder', 'Milk_Soy Milk', 'Milk_Anmuxi', 'Instant Noodles_Haidilao Tomato Beef Soup Noodles', 'Snacks_Haoyouqu Potato Chips Bagged', 'Candy_Animal Gummies', 'Seasoning_Li Salt', 'Milk_Coffee Milk', 'Chewing Gum_Doublemint Lozenges', 'Dessert_Chip Delight Soft Cookies', 'Condiment_Qishen Horseradish', 'Chocolate_Kinder Chocolate', 'Beverage_Red Bull', 'Candy_Skittles Bag', 'Stationery_Scissors', 'Dessert_Quduo Mini Cookies', 'Instant Drink_Nescafé Absolute Deep Black', 'Canned_Fish Roe Caviar', "Puffed Snacks_Lay's Potato Chips Canister", 'Stationery_Gel Pen', 'Snacks_Ya Potato', 'Dried Fruit_Qiaqia Hi Sunflower Seeds', 'Milk_Mengniu Boxed', 'Dried Fruit_Strange Flavor Beans', 'Instant Noodles_Lucky Bear Spicy Sour Powder', 'Alcohol_Tiger Beer', 'Dried Fruit_Dried Tangerine Peel', 'Stationery_Notebook', 'Chocolate_JiuJiu Dark Chocolate', 'Personal Care_Shanghai Medicinal Soap', 'Dried Fruit_Green Peas', 'Candy_Popping Candy', 'Dried Fruits_Qiaqia Caramel Melon Seeds', 'Chocolate_Mylikes', 'Milk_Wangzai Milk Can', 'Dried Fruit_Castanets', 'Stationery_Eraser', 'Instant Noodles_Spicy Beef Noodles', 'Chocolate_Mylikes Coconut Flavor', 'Dessert_Oreo Chocolate Flavor', 'Instant Drink_Yo-lo-mei', 'Chewing Gum_Mentos', 'Dried Fruit_Yangmei', 'Snacks_Friend Fun Steak Flavored Chips', 'Condiment_Qishen Black Pepper Powder', 'Alcohol_Snow Beer Blue', 'Candy_Orion Orange Candy', 'Instant Noodles_Spicy Beef Noodles', 'Instant Noodles_Tomato Egg Beef Noodles', 'Instant Noodles_Shrimp Fish Cake Noodles', 'Dessert_Chocolate Pie', 'Drink_Coconut Tree Coconut Juice', 'Drink_Sprite', 'Drink_Coca-Cola', 'Drink_Suntory Oolong Tea', 'Drink_Enen Soda Water', 'Dessert_Oreo Original Flavor', 'Canned_Fresh Stewed Tremella by Xiduoduo', 'Dessert_Q Ti', 'Chocolate_Green Chocolate', 'Tissue_Banbu Facial Tissue', "Snacks_Lay's Barbecue Flavored Chips", 'Milk_AD Calcium Milk', 'Beverage_Dalian Soda', 'Personal Care_Tissues', 'Stationery_Correction Tape', 'Dry Fruits_Mustard Peanuts'),

        # ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        # 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        # 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        # 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        # 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        # 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        # 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        # 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        # 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        # 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        # 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        # 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        # 'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
         (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
         (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
         (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
         (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
         (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
         (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
         (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
         (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
         (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
         (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
         (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
         (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
         (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
         (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
         (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
         (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
         (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
         (246, 0, 122), (191, 162, 208)]
    }
    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['custom_entities'] = True

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

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
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
        data_info['instances'] = instances
        return data_info

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        # obtain images that contain annotation
        ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos
