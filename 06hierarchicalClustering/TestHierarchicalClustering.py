import os
from PIL import Image,ImageDraw
import numpy as np
from HierarchicalClustering import hcluster, getheight, getdepth

# ---------------------------------------------
# 生成图片
def drawdendrogram(clust, imlist, jpeg='clusters.jpg'):
    # 获得高度和宽度
    h = getheight(clust)*20
    w = 1200
    depth = getdepth(clust)

    # 由于宽度固定，因此需要对距离进行调整
    scaling = float(w-150)/depth

    # 创建白色背景的图片
    img = Image.new('RGB', (w,h), (255,255,255))
    draw = ImageDraw.Draw(img)

    draw.line((0,h/2,10,h/2), fill=(255,0,0))

    # 画第一个节点
    drawnode(draw, clust, 10, (h/2), scaling, imlist, img)
    img.save(jpeg, 'JPEG')


#----------------------------------------------------------------------
# 画节点和连线
def drawnode(draw, clust, x, y, scaling, imlist, img):
    if clust.id < 0:
        h1 = getheight(clust.left)*20
        h2 = getheight(clust.right)*20

        top = y - (h1+h2)/2
        bottom = y + (h1+h2)/2

        # 线的长度
        l1 = clust.distance * scaling
        # 聚类到其子节点的垂直线
        draw.line((x, top + h1/2, x, bottom - h2/2), fill=(255,0,0))

        # 连接左侧节点的水平线
        draw.line((x, top+h1/2, x+l1, top+h1/2), fill=(255,0,0))

        # 连接右侧节点的水平线
        draw.line((x, bottom-h2/2, x+l1, bottom-h2/2), fill=(255,0,0))

        # 绘制左右节点
        drawnode(draw, clust.left, x+l1, top+h1/2, scaling, imlist, img)
        drawnode(draw, clust.right, x+l1, bottom-h2/2, scaling, imlist, img)
    else:
        # 是叶节点则绘制其标签
        nodeim = Image.open(imlist[clust.id])
        nodeim.thumbnail((20, 20))
        ns = nodeim.size
        print(x, y-ns[1]//2)
        print(x+ns[0])
        print(img.paste(nodeim, (int(x), int(y-ns[1]//2), int(x+ns[0]), int(y+ns[1]-ns[1]//2))))

if __name__ == '__main__':
    # 创建图片列表
    imlist = []
    folderPath = r'D:\照片\宝宝6岁写真照min'    # 需要缓冲电脑中一个图片文件夹的路径
    for filename in os.listdir(folderPath):
        if os.path.splitext(filename)[1] == '.jpg':
            imlist.append(os.path.join(folderPath, filename))
    n = len(imlist)
    print(n)

    # 获取每张图片的特征向量RGB
    features = np.zeros((n,3))
    for i in range(n):
        im = np.array(Image.open(imlist[i]))
        R = np.mean(im[:,:,0].flatten())
        G = np.mean(im[:,:,1].flatten())
        B = np.mean(im[:,:,2].flatten())
        features[i] = np.array([R,G,B])

    tree = hcluster(features)
    drawdendrogram(tree, imlist, jpeg='myHClusing.jpg')

