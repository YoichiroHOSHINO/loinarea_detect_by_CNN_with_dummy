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


# 矩形回転関数
def rotatedRectangle(img, rotatedRect, color, thickness=1, lineType=cv2.LINE_8, shift=0):
    (x,y), (width, height), angle = rotatedRect
    angle = math.radians(angle)
 
    # 回転する前の矩形の頂点
    pt1_1 = (int(x + width / 2), int(y + height / 2))
    pt2_1 = (int(x + width / 2), int(y - height / 2))
    pt3_1 = (int(x - width / 2), int(y - height / 2))
    pt4_1 = (int(x - width / 2), int(y + height / 2))
 
    # 変換行列
    t = np.array([[np.cos(angle),   -np.sin(angle), x-x*np.cos(angle)+y*np.sin(angle)],
                    [np.sin(angle), np.cos(angle),  y-x*np.sin(angle)-y*np.cos(angle)],
                    [0,             0,              1]])
 
    tmp_pt1_1 = np.array([[pt1_1[0]], [pt1_1[1]], [1]])
    tmp_pt1_2 = np.dot(t, tmp_pt1_1)
    pt1_2 = (int(tmp_pt1_2[0][0]), int(tmp_pt1_2[1][0]))
 
    tmp_pt2_1 = np.array([[pt2_1[0]], [pt2_1[1]], [1]])
    tmp_pt2_2 = np.dot(t, tmp_pt2_1)
    pt2_2 = (int(tmp_pt2_2[0][0]), int(tmp_pt2_2[1][0]))
 
    tmp_pt3_1 = np.array([[pt3_1[0]], [pt3_1[1]], [1]])
    tmp_pt3_2 = np.dot(t, tmp_pt3_1)
    pt3_2 = (int(tmp_pt3_2[0][0]), int(tmp_pt3_2[1][0]))
 
    tmp_pt4_1 = np.array([[pt4_1[0]], [pt4_1[1]], [1]])
    tmp_pt4_2 = np.dot(t, tmp_pt4_1)
    pt4_2 = (int(tmp_pt4_2[0][0]), int(tmp_pt4_2[1][0]))
 
    points = np.array([pt1_2, pt2_2, pt3_2, pt4_2])
    cv2.polylines(img, [points], True, color, thickness, lineType, shift)
    return img
 
img = np.zeros((300, 300, 3), np.uint8)
rotatedRect = ((150, 150), (200, 120), 20)
rotatedRectangle(img, rotatedRect, (255, 255, 0))
 
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
