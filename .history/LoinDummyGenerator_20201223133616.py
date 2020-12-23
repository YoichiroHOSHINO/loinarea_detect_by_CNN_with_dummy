#%%
import cv2
import numpy as np
from numpy.random import *
from tqdm import tqdm 

# 画像サイズ指定
height=300
width=300

# 疑似脂肪交雑レート

for i in tqdm(range(10)):
    # 脂肪（白）領域
    # 中心を中心x,y±10%の範囲でランダムに決定。
    outcenterx = width/2 + rand()*width/10 - width/20
    outcentery = height/2 + rand()*height/10 - height/20
    # 領域のv,hサイズを200～250の範囲でランダムに決定。
    outrangev = rand()*50 + 200
    outrangeh = rand()*50 + 200
    # 領域を±90度の範囲で回転。
    outangle = rand()*180-90

    # ロース芯を２つの楕円で表現
    # ロース芯１の中心座標xを脂肪領域中心－0～50で決定。
    loin1centerx = outcenterx - rand()*50
    # ロース芯１の中心座標yを脂肪領域中心±25で決定。
    loin1centery = outcentery + rand()*50 - 25
    # ロース芯１の領域hを脂肪領域外縁に接する円x0.9～1.0倍に決定。外縁に触れるか触れないかの大きさに。
    loin1rangeh = (outrangeh/2 - (outcenterx - loin1centerx))*2*(0.9 + rand()*0.1)
    # ロース芯１の領域vをhの0.9～1.1倍に決定。
    loin1rangev = loin1rangeh * (0.9 + rand()*0.2)
    # ロース芯１を±90度の範囲で回転。
    loin1angle = rand()*180-90

    # ロース芯２の中心座標xを脂肪領域中心＋0～50で決定。
    loin2centerx = outcenterx + rand()*50
    # ロース芯２の中心座標yを脂肪領域中心±25で決定。
    loin2centery = outcentery + rand()*50 - 25
    # ロース芯２の領域hを脂肪領域外縁に接する円x0.9～1.0倍に決定。外縁に触れるか触れないかの大きさに。
    loin2rangeh = (outrangeh/2 - (loin2centerx - outcenterx))*2*(0.9 + rand()*0.1)
    # ロース芯２の領域vをhの0.9～1.1倍に決定。
    loin2rangev = loin2rangeh * (0.9 + rand()*0.2)
    # ロース芯２を±90度の範囲で回転。
    loin2angle = rand()*180-90

    # 疑似脂肪交雑レート
    marblerate = rand()*0.5

    # 楕円描画
    img = np.full((height, width, 1), np.float(255), dtype=np.uint8)
    img = cv2.ellipse(img, ((outcenterx,outcentery), (outrangev, outrangeh), outangle), np.float(0), thickness=-1)
    img = cv2.ellipse(img, ((loin1centerx,loin1centery), (loin1rangev, loin1rangeh), loin1angle), np.float(255), thickness=-1)
    img = cv2.ellipse(img, ((loin2centerx,loin2centery), (loin2rangev, loin2rangeh), loin2angle), np.float(255), thickness=-1)

    marble = rand(width, height, 1)
    marble = np.where(marble <= marblerate, np.float(0), np.float(1))
    img = img * marble
    img = 255 - img

    bwimg = np.full((height, width, 1), np.float(0), dtype=np.uint8)
    bwimg = cv2.ellipse(bwimg, ((loin1centerx,loin1centery), (loin1rangev, loin1rangeh), loin1angle), 255, thickness=-1)
    bwimg = cv2.ellipse(bwimg, ((loin2centerx,loin2centery), (loin2rangev, loin2rangeh), loin2angle), 255, thickness=-1)
    
    cv2.imwrite('./img/demo_'+str(i)+'.bmp', img)
    cv2.imwrite('./bwimg/mask_'+str(i)+'.bmp', bwimg)

