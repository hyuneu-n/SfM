import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch

from models.superpoint import SuperPoint
from models.superglue import SuperGlue
from models.utils import frame2tensor


# 디바이스 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# SuperPoint 및 SuperGlue 모델 로드
def load_superpoint_superglue_models():
    # SuperPoint 설정
    superpoint_config = {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1
    }
    superpoint = SuperPoint(superpoint_config).to(device).eval()

    # SuperGlue 설정
    superglue_config = {
        'weights': 'indoor',  # 이미지에 따라 'outdoor'로 변경 가능
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2
    }
    superglue = SuperGlue(superglue_config).to(device).eval()

    return superpoint, superglue

superpoint, superglue = load_superpoint_superglue_models()

# 이미지 경로 설정
img_path = '/home/hyuneun/disk_b/ESL/SfM/sfm/data/nutellar2/'

img1_name = 'nutella4.jpg'
img2_name = 'nutella5.jpg'
img3_name = 'nutella6.jpg'
img4_name = 'nutella7.jpg'

# 이미지 대비 조절 함수
def adjust_contrast(img, alpha=1.2, beta=20):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

# 이미지 로드 함수
def load_image(img_path, img1_name, img2_name, contrast=False):
    img1 = cv2.imread(img_path + img1_name)
    img2 = cv2.imread(img_path + img2_name)
    
    if img1 is None:
        raise FileNotFoundError(f"Error: Unable to load image {img1_name} from {img_path}")
    if img2 is None:
        raise FileNotFoundError(f"Error: Unable to load image {img2_name} from {img_path}")
    
    # 대비 조절 (원할 경우)
    if contrast:
        img1 = adjust_contrast(img1)
        img2 = adjust_contrast(img2)
    
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    return img1, img2

# SuperPoint와 SuperGlue를 사용한 특징점 추출 및 매칭 함수
def SuperPoint_SuperGlue(img1, img2, img1_name, img2_name):
    # 이미지를 그레이스케일로 변환
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # 이미지를 텐서로 변환
    img1_tensor = frame2tensor(img1_gray, device)
    img2_tensor = frame2tensor(img2_gray, device)

    # SuperPoint로 특징점 추출
    with torch.no_grad():
        pred1 = superpoint({'image': img1_tensor})
        pred2 = superpoint({'image': img2_tensor})

    keypoints0 = pred1['keypoints'][0].cpu().numpy()
    descriptors0 = pred1['descriptors'][0].cpu().numpy()
    keypoints1 = pred2['keypoints'][0].cpu().numpy()
    descriptors1 = pred2['descriptors'][0].cpu().numpy()

    # 임의의 scores0, scores1을 추가 (SuperPoint에서 스코어가 없으므로 기본값으로 설정)
    scores0 = torch.ones(keypoints0.shape[0]).unsqueeze(0).to(device)  # 크기 맞춰 임의의 스코어 설정
    scores1 = torch.ones(keypoints1.shape[0]).unsqueeze(0).to(device)  # 크기 맞춰 임의의 스코어 설정
    

    # SuperGlue를 위한 데이터 준비
    input_data = {
        'keypoints0': torch.tensor(keypoints0).unsqueeze(0).to(device),
        'keypoints1': torch.tensor(keypoints1).unsqueeze(0).to(device),
        'descriptors0': torch.tensor(descriptors0).unsqueeze(0).to(device),
        'descriptors1': torch.tensor(descriptors1).unsqueeze(0).to(device),
        'scores0': scores0,  # 추가된 scores0
        'scores1': scores1,  # 추가된 scores1
        'image0': img1_tensor,
        'image1': img2_tensor,
    }


    # SuperGlue로 매칭 수행
    with torch.no_grad():
        pred = superglue(input_data)

    matches = pred['matches0'][0].cpu().numpy()
    confidence = pred['matching_scores0'][0].cpu().numpy()

    # 매칭된 특징점 추출
    valid = matches > -1
    mkpts0 = keypoints0[valid]
    mkpts1 = keypoints1[matches[valid]]
    mconf = confidence[valid]

    # cv2.KeyPoint 객체 생성
    img1_kp = [cv2.KeyPoint(x=float(kp[0]), y=float(kp[1]), size=1) for kp in keypoints0]
    img2_kp = [cv2.KeyPoint(x=float(kp[0]), y=float(kp[1]), size=1) for kp in keypoints1]

    # cv2.DMatch 객체 생성
    matches_good = []
    for idx, (queryIdx, trainIdx, conf) in enumerate(zip(np.where(valid)[0], matches[valid], mconf)):
        match = cv2.DMatch(_queryIdx=queryIdx, _trainIdx=trainIdx, _imgIdx=0, _distance=1 - conf)
        matches_good.append(match)

    # 매칭 결과 시각화
    res = cv2.drawMatches(img1, img1_kp, img2, img2_kp, matches_good, None, flags=2)

    # 매칭 결과 시각화
    plt.figure(figsize=(15, 15))
    plt.imshow(res)
    plt.title(f'Matching: {img1_name} & {img2_name}')
    plt.show()

    # 매칭된 이미지 파일로 저장
    matched_image_path = f"matched_{img1_name}_vs_{img2_name}.png"
    plt.imsave(matched_image_path, res)
    print(f"매칭된 이미지 저장 완료: {matched_image_path}")

    return matches_good, img1_kp, img2_kp

# 에센셜 매트릭스 추정
def Estimation_E(matches_good, img1_kp, img2_kp):
    query_idx = [match.queryIdx for match in matches_good]
    train_idx = [match.trainIdx for match in matches_good]
    p1 = np.float32([img1_kp[ind].pt for ind in query_idx]) 
    p2 = np.float32([img2_kp[ind].pt for ind in train_idx])

    E, mask = cv2.findEssentialMat(p1, p2, method=cv2.RANSAC, focal=3092.8, pp=(2016, 1512), maxIters=1000, threshold=0.3)
    
    p1_inlier = p1[mask.ravel() == 1]
    p2_inlier = p2[mask.ravel() == 1]

    return E, p1_inlier, p2_inlier

# 에센셜 매트릭스 분해
def EM_Decomposition(E, p1, p2):
    U, S, VT = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(VT) < 0:
        VT *= -1

    camera_matrix_options = [
        np.column_stack((U @ W @ VT, U[:, 2])),
        np.column_stack((U @ W @ VT, -U[:, 2])),
        np.column_stack((U @ W.T @ VT, U[:, 2])),
        np.column_stack((U @ W.T @ VT, -U[:, 2]))
    ]

    # 카메라 매트릭스 선택 (여기서는 첫 번째 사용)
    return camera_matrix_options[0]

# 내부 카메라 행렬 초기화
def initialize_CM(CameraMatrix):
    Rt0 = np.hstack((np.eye(3), np.zeros((3, 1))))
    skew = 0.0  # 왜곡 계수는 0으로 설정
    K = np.array([[3092.8, skew, 2016], [0, 3092.8, 1512], [0, 0, 1]])
    Rt1 = K @ CameraMatrix
    return Rt0, Rt1

# 삼각측량
def LinearTriangulation(Rt0, Rt1, p1, p2):
    A = np.array([
        p1[1] * Rt0[2, :] - Rt0[1, :],
        p1[0] * Rt0[2, :] - Rt0[0, :],
        p2[1] * Rt1[2, :] - Rt1[1, :],
        p2[0] * Rt1[2, :] - Rt1[0, :]
    ])

    _, _, VT = np.linalg.svd(A)
    X = VT[-1]
    return X[0:3] / X[3]

# 3D 포인트 생성
def make_3dpoint(p1, p2, Rt0, Rt1):
    p3ds = []
    for pt1, pt2 in zip(p1, p2):
        p3d = LinearTriangulation(Rt0, Rt1, pt1, pt2)
        p3ds.append(p3d)
    return np.array(p3ds).T

# 3D 시각화
def visualize_3d(p3ds, filename="3d_plot.png"):
    X = p3ds[0]
    Y = p3ds[1]
    Z = p3ds[2]

    fig = plt.figure(figsize=(15, 15))
    ax = plt.axes(projection='3d')
    ax.scatter3D(X, Y, Z, c='b', marker='o') 
    plt.savefig(filename)  # 이미지를 파일로 저장
    print(f"3D 시각화 저장 완료: {filename}")

# 좌표계 맞춤
def align_coordinate_system(Rt1_first, Rt1_second):
    R_first = Rt1_first[:, :3]
    R_second = Rt1_second[:, :3]
    T = np.linalg.inv(R_first) @ R_second
    return T

# 2-view 재구성 함수
def reconstruct_2view(img1_name, img2_name, view_id):
    img1, img2 = load_image(img_path, img1_name, img2_name)
    matches_good, img1_kp, img2_kp = SuperPoint_SuperGlue(img1, img2, img1_name, img2_name)
    E, p1_inlier, p2_inlier = Estimation_E(matches_good, img1_kp, img2_kp)
    CameraMatrix = EM_Decomposition(E, p1_inlier, p2_inlier)
    Rt0, Rt1 = initialize_CM(CameraMatrix)
    point3d = make_3dpoint(p1_inlier, p2_inlier, Rt0, Rt1)
    
    # 각각의 2-view에 대해 3D 시각화 저장
    visualize_3d(point3d, f"3d_plot_view_{view_id}.png")  # view_id로 구분된 파일명 저장
    print(f"3D 포인트 클라우드 시각화 저장 완료: 3d_plot_view_{view_id}.png")
    
    return point3d, Rt1

# 두 2-view 결합
def reconstruct_2sets():
    p3ds_12, Rt1_first = reconstruct_2view(img1_name, img2_name, 1)
    p3ds_34, Rt1_second = reconstruct_2view(img3_name, img4_name, 2)
    
    # 좌표계 맞춤
    T = align_coordinate_system(Rt1_first, Rt1_second)
    p3ds_34_transformed = T @ p3ds_34
    
    # 두 2-view 결과를 합치기
    p3ds_combined = np.hstack((p3ds_12, p3ds_34_transformed))

    # 최종 3D 시각화
    print("3D 포인트 계산 완료, 최종 시각화 시작")
    visualize_3d(p3ds_combined, "3d_plot_combined.png")

# 2세트의 2-view 재구성 실행
reconstruct_2sets()
