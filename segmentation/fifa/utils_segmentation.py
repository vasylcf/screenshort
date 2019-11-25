import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import matplotlib.pyplot as plt
import cv2


def create_annotations(FF,img_folder, annotation_folder):
    """
    Create annotations for images
    Input:
    FF - data_frame with columns:  'fname','labels','bbox_x1','bbox_y1','bbox_x2','bbox_y2','bbox_x3','bbox_y3','bbox_x4','bbox_y4'
    """
    for f in FF.fname.unique():
        img = cv2.imread(img_folder+f)
        print(img_folder+f)
        mask = np.zeros(img.shape)
        T = FF[FF.fname==f]
        for cl in T['labels']:
            pts = T[T['labels']==cl][['bbox_x1','bbox_y1','bbox_x2','bbox_y2','bbox_x3','bbox_y3','bbox_x4','bbox_y4']].values[0].reshape(4,2)
            pts = pts.astype('int')
            mask = cv2.fillPoly(mask,[pts],color=(cl,cl,cl))
        cv2.imwrite(annotation_folder+f.split('.')[0]+'.png',mask)


def augment_seg(img,seg,seq):
    """
    for augmentation
    """
    aug_det = seq.to_deterministic()
    image_aug = aug_det.augment_image(img)
    segmap = ia.SegmentationMapOnImage(seg, nb_classes=np.max(seg)+1, shape=img.shape)
    segmap_aug = aug_det.augment_segmentation_maps(segmap)
    segmap_aug = segmap_aug.get_arr_int()
    return image_aug, segmap_aug

def augmentation_images(images,annotations):
    """
    images = glob.glob('images_train_2/*.png')
    annotations = glob.glob('annotations_train_2/*.png')
    """
    
    img_path = images[0].split('/')[0]
    annot_path = annotations[0].split('/')[0]
    for i,s in zip(images,annotations):
    
        fname = i.split('/')[-1].split('.')[0]
        img = cv2.imread(i)
        seg = cv2.imread(s)

        for n in range(20):

            # crop augmentation
            bnds = np.random.randint(5,40,4)
            seq_crop = iaa.Sequential([iaa.Crop(px=tuple(bnds))])
            img_aug, seg_aug = augment_seg(img,seg,seq_crop)
            cv2.imwrite( img_path + '/'+fname+'_crop'+str(n)+'.png',img_aug)
            cv2.imwrite(annot_path+'/'+fname+'_crop'+str(n)+'.png',seg_aug)

            # rotate augmentation
            seq_rotate = iaa.Sequential([iaa.Affine(rotate=(-10,10))])
            img_aug, seg_aug = augment_seg(img,seg,seq_rotate)
            cv2.imwrite(img_path+'/'+fname+'_rotate'+str(n)+'.png',img_aug)
            cv2.imwrite(annot_path+'/'+fname+'_rotate'+str(n)+'.png',seg_aug)

            # scale augmentation
            seq_scale = iaa.Sequential([iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)})])
            img_aug, seg_aug = augment_seg(img,seg,seq_scale)
            cv2.imwrite(img_path+'/'+fname+'_scale'+str(n)+'.png',img_aug)
            cv2.imwrite(annot_path+'/'+fname+'_scale'+str(n)+'.png',seg_aug)

            # blur augmentation
            seq_blur = iaa.Sequential([iaa.GaussianBlur(sigma=(0, 3.0))])
            img_aug, seg_aug = augment_seg(img,seg,seq_blur)
            cv2.imwrite(img_path+'/'+fname+'_blur'+str(n)+'.png',img_aug)
            cv2.imwrite(annot_path+'/'+fname+'_blur'+str(n)+'.png',seg_aug)

def find_segments(mask_img,cls):
    """
    Find segments on mask
    Inputs:
    mask_img - segmentation mask image from model
    cls - labels of classes that are on image
    Outputs:
    segments - segmnets that were found

    """
    segments={}
    for cl in cls:
        cols = np.sum(mask_img==cl,axis=1)
        rows = np.sum(mask_img==cl,axis=0)
        col_cond1= (cols>=np.mean(cols[cols>4])/1.5)
        #col_cond2 = (cols<np.mean(cols[cols>4])*1.5)
        row_cond1= (rows>=np.mean(rows[rows>4])/1.5)
        #row_cond2 = (rows<np.mean(rows[rows>4])*1.5)
        col_indecies = np.where(col_cond1)[0]
        row_indecies = np.where(row_cond1)[0]

        x1 = row_indecies[0]
        y1 = col_indecies[0]
        x2 = row_indecies[-1]
        y2 = col_indecies[-1]
        
        segments[cl] = [(x1,y1),(x2,y2)]
    return segments

def crop_rect(im, rect, scaler={'h': 1, 'w': 1}):
    """
    Crop original image by segments.
    Input:
    im - raw image
    rect - rectangle for cropping
    scaler - scaler for rectangle
    Return:
    crop_img - crop segment with original resolution
    coordinates - coordinates of crop rectangle
    """
    h1 = int(rect[0][1]*scaler['h'])
    w1 = int(rect[0][0]*scaler['w'])
    h2 = int(rect[1][1]*scaler['h'])
    w2 = int(rect[1][0]*scaler['w'])
    crop_img = im[h1:h2, w1:w2]
    #[(x1,y1),(x2,y2)]
    coordinates = [(w1,h1),(w2,h2)]        
    return crop_img, coordinates

def detect_classes(img,n_classes):
    """
    Detect classes that are most represemted on mask
    """
    labels, counts = np.unique(img,return_counts=True)
    labels = labels[1:]
    counts = counts[1:]
    inds = np.argsort(counts)
    list_classes = []
    for i in range(1,n_classes+1):
        list_classes.append(labels[inds[-i]])
    return list_classes


def build_hist(mask_img, labels):
    """
    Building histogram for mask by classes
    """
    segments={}
    for cl in labels:
        cols = np.sum(mask_img==cl,axis=0)
        rows = np.sum(mask_img==cl,axis=1)
        segments[cl]={'cols':cols, 'rows':rows}
    
    r=len(labels)
    c=3
    k=1
    plt.figure(figsize=(14,8))
    for i in segments:
        plt.subplot(r,c,k)
        plt.title(f'Label:{i},rows')
        plt.axis([0,112,0,112])
        plt.grid()
        ax=plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_aspect('equal')
        plt.barh(np.arange(1,113),segments[i]['rows'])
        k+=1
        plt.subplot(r,c,k)
        plt.title(f'Label:{i},cols')
        plt.bar(np.arange(1,113),segments[i]['cols'])
        plt.axis([0,112,0,112])
        plt.grid()
        ax=plt.gca()
        ax.set_aspect('equal')
        ax.set_ylim(ax.get_ylim()[::-1])
        k+=1
        plt.subplot(r,c,k)
        plt.title(f'mask')
        plt.imshow(mask_img)
        k+=1

def analysis(f, model, path='test_labeled_dataset/data/', size=(224,224), n_classes=2):
    """
    Visualization of work of model
    """

    img = cv2.imread(path+f)
    img_r = cv2.resize(img, size)
    img_r_mask = cv2.resize(img, (int(size[0]/2),int(size[1]/2)))

    out = model.predict_segmentation(img_r)

    # h_scale = int(np.round(img.shape[0] / out.shape[0], 0))
    # w_scale = int(np.round(img.shape[1] / out.shape[1], 0))
    h_scale = img.shape[0] / out.shape[0]
    w_scale = img.shape[1] / out.shape[1]
    scaler = {'h': h_scale, 'w': w_scale, }

    list_cls = detect_classes(out,n_classes)
    rects = find_segments(out,list_cls)
    segments = []
    for r in rects:
        crop_img, coordinates = crop_rect(img, rects[r], scaler=scaler)
        segments.append({'label': r, 'data': crop_img, 'coordinates':coordinates})
        
        cv2.rectangle(img_r_mask,*rects[r],(255,255,0),2)

    print(f'Find segments: {len(segments)}')
    plt.figure(figsize=(14,6))
    plt.subplot(1,4,1)
    plt.title(f"Label: {segments[0]['label']}")
    plt.imshow(segments[0]['data'])
    plt.subplot(1,4,2)
    plt.title(f"Label: {segments[1]['label']}")
    plt.imshow(segments[1]['data'])
    plt.subplot(1,4,3)
    plt.title(f"mask")
    plt.imshow(out)
    plt.subplot(1,4,4)
    plt.title(f"image")
    plt.imshow(img_r_mask)
    return out

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    # return the intersection over union value
    return iou

def IOU_list_files(test_data,model,img_folder='test_labeled_dataset/data/',
                   annotation_folder = 'test_labeled_dataset/annotations_test_stat_segm/',n_classes=2,
                  size=(224,224)):

    """
    For testing accuracy of model. I use metric IOU.
    Input:
    test_data - data frame with columns:  'fname','labels','bbox_x1','bbox_y1','bbox_x2','bbox_y2','bbox_x3','bbox_y3','bbox_x4','bbox_y4'
    img_folder - folder with test images

    """
    m=np.array([[True,True],[False,False],[True,True],[False,False]])
    
    predicted_boxes = {}
    real_boxes = {}
    IOUs = {}

    for f in test_data.fname.unique():

        img = cv2.imread(img_folder+f)
        img_r = cv2.resize(img, size)

        out = model.predict_segmentation(img_r)
        h_scale = np.round(img.shape[0] / out.shape[0], 0)
        w_scale = np.round(img.shape[1] / out.shape[1], 0)
        scaler = {'h': h_scale, 'w': w_scale, }
        list_cls = detect_classes(out,n_classes)
        rects = find_segments(out,list_cls)
        segments = {}
        for r in rects:
            crop_img, coordinates = crop_rect(img, rects[r], scaler=scaler)
            segments[r]={'data': crop_img, 'coordinates':coordinates}

        predicted_boxes[f] = segments

        real_boxes[f]={}
        IOUs[f] = {}

        for l in test_data[test_data.fname==f].labels:
            pts=test_data[(test_data.fname==f)&(test_data.labels==l)].iloc[:,:-3].values[0].astype('int').reshape(4,2)
            real_boxes[f][l] = pts[m]

            boxA= np.ravel(segments[l]['coordinates'])
            boxB = pts[m]

            IOUs[f][l] = bb_intersection_over_union(boxA, boxB)
            
    return IOUs