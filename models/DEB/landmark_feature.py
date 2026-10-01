import torch


class LandmarkFeatureExtractor:
    @staticmethod
    def calculate_angle(A, B, C):
        BA = A - B
        BC = C - B
        dot_product = torch.sum(BA * BC, dim=1)
        magnitude_BA = torch.norm(BA, dim=1)
        magnitude_BC = torch.norm(BC, dim=1)
        cos_theta = dot_product / (magnitude_BA * magnitude_BC)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        angle_rad = torch.acos(cos_theta)
        angle_deg = angle_rad * 180.0 / torch.pi

        return angle_deg

    @staticmethod
    def calculate_distance(p1, p2):
        squared_diff = (p2 - p1) ** 2
        sum_squared_diff = torch.sum(squared_diff, dim=1)
        distances = torch.sqrt(sum_squared_diff)
        return distances

    @staticmethod
    def landmarkfeature1(landmarks_tensor, left_iris_tensor, right_iris_tensor):
        p20, p25, p28 = landmarks_tensor[:, 19], landmarks_tensor[:, 24], landmarks_tensor[:, 27]
        angle1 = LandmarkFeatureExtractor.calculate_angle(p20, p28, left_iris_tensor)
        angle2 = LandmarkFeatureExtractor.calculate_angle(p25, p28, right_iris_tensor)
        return abs(angle1 - angle2)

    @staticmethod
    def landmarkfeature2(landmarks_tensor):
        p40, p18, p43, p27, p37, p46 = landmarks_tensor[:, 39], landmarks_tensor[:, 17], landmarks_tensor[:, 42], landmarks_tensor[:, 26], landmarks_tensor[:, 36], landmarks_tensor[:, 45]
        angle1 = LandmarkFeatureExtractor.calculate_angle(p18, p40, p37)
        angle2 = LandmarkFeatureExtractor.calculate_angle(p27, p43, p46)
        return abs(angle1 - angle2)

    @staticmethod
    def landmarkfeature3(landmarks_tensor):
        p34, p49, p55 = landmarks_tensor[:, 33], landmarks_tensor[:, 48], landmarks_tensor[:, 54]
        angle1 = LandmarkFeatureExtractor.calculate_angle(p34, p49, p55)
        angle2 = LandmarkFeatureExtractor.calculate_angle(p34, p55, p49)
        return abs(angle1 - angle2)

    @staticmethod
    def landmarkfeature4(landmarks_tensor, left_iris_tensor, right_iris_tensor):
        p49, p55, p58 = landmarks_tensor[:, 48], landmarks_tensor[:, 54], landmarks_tensor[:, 57]
        angle1 = LandmarkFeatureExtractor.calculate_angle(left_iris_tensor, p49, p58)
        angle2 = LandmarkFeatureExtractor.calculate_angle(right_iris_tensor, p55, p58)
        return abs(angle1 - angle2)

    @staticmethod
    def landmarkfeature5(landmarks_tensor):
        left_eye_indices = [36, 37, 38, 39, 40, 41]
        right_eye_indices = [45, 44, 43, 42, 47, 46]

        left_eye_points = landmarks_tensor[:, left_eye_indices]
        right_eye_points = landmarks_tensor[:, right_eye_indices]

        left_eye_distances = [
            LandmarkFeatureExtractor.calculate_distance(left_eye_points[:, 1], left_eye_points[:, 5]),
            LandmarkFeatureExtractor.calculate_distance(left_eye_points[:, 2], left_eye_points[:, 4]),
            LandmarkFeatureExtractor.calculate_distance(left_eye_points[:, 0], left_eye_points[:, 3])
        ]
        right_eye_distances = [
            LandmarkFeatureExtractor.calculate_distance(right_eye_points[:, 1], right_eye_points[:, 5]),
            LandmarkFeatureExtractor.calculate_distance(right_eye_points[:, 2], right_eye_points[:, 4]),
            LandmarkFeatureExtractor.calculate_distance(right_eye_points[:, 0], right_eye_points[:, 3])
        ]

        distance1 = (left_eye_distances[0] + left_eye_distances[1]) / (2 * left_eye_distances[2])
        distance2 = (right_eye_distances[0] + right_eye_distances[1]) / (2 * right_eye_distances[2])
        return abs(distance1 - distance2)

    @staticmethod
    def extract_features(landmarks_tensor, left_iris_tensor, right_iris_tensor):
        B, T = landmarks_tensor.shape[:2]
        landmarks_flat = landmarks_tensor.view(B * T, 68, 2)
        left_iris_flat = left_iris_tensor.view(B * T, 2)
        right_iris_flat = right_iris_tensor.view(B * T, 2)

        f1 = LandmarkFeatureExtractor.landmarkfeature1(landmarks_flat, left_iris_flat, right_iris_flat)
        f2 = LandmarkFeatureExtractor.landmarkfeature2(landmarks_flat)
        f3 = LandmarkFeatureExtractor.landmarkfeature3(landmarks_flat)
        f4 = LandmarkFeatureExtractor.landmarkfeature4(landmarks_flat, left_iris_flat, right_iris_flat)
        f5 = LandmarkFeatureExtractor.landmarkfeature5(landmarks_flat)

        features = torch.stack([f1, f2, f3, f4, f5], dim=1).view(B, T, -1)

        return features


class LandmarkFeatureExtractor1:

    @staticmethod
    def correct_landmarks(landmarks, left_iris, right_iris):
        batch_size, seq_len = landmarks.shape[:2]
        landmarks = landmarks.reshape(-1, 68, 2)
        point_a = landmarks[:, 0]
        point_b = landmarks[:, 16]

        delta_x = point_b[:, 0] - point_a[:, 0]
        delta_y = point_b[:, 1] - point_a[:, 1]
        angle = torch.atan2(delta_y, delta_x)

        cos_theta = torch.cos(angle)
        sin_theta = torch.sin(angle)

        center_x, center_y = point_a[:, 0], point_a[:, 1]

        zeros = torch.zeros_like(cos_theta)
        ones = torch.ones_like(cos_theta)

        rotation_matrix = torch.stack([
            torch.stack([cos_theta, sin_theta, center_x * (1 - cos_theta) - center_y * sin_theta], dim=1),
            torch.stack([-sin_theta, cos_theta, center_y * (1 - cos_theta) + center_x * sin_theta], dim=1),
            torch.stack([zeros, zeros, ones], dim=1)
        ], dim=2)

        ones = torch.ones(*landmarks.shape[:-1], 1, device=landmarks.device)
        landmarks_homogeneous = torch.cat([landmarks, ones], dim=-1)

        left_iris = left_iris.reshape(-1, 2).unsqueeze(1)
        right_iris = right_iris.reshape(-1, 2).unsqueeze(1)

        iris_ones = torch.ones(*left_iris.shape[:-1], 1, device=left_iris.device)
        left_iris_homogeneous = torch.cat([left_iris, iris_ones], dim=-1)
        right_iris_homogeneous = torch.cat([right_iris, iris_ones], dim=-1)

        rotated_landmarks = torch.bmm(landmarks_homogeneous, rotation_matrix)
        rotated_left_iris = torch.bmm(left_iris_homogeneous, rotation_matrix)
        rotated_right_iris = torch.bmm(right_iris_homogeneous, rotation_matrix)

        rotated_landmarks = rotated_landmarks[..., :2].reshape(batch_size, seq_len, 68, 2)
        rotated_left_iris = rotated_left_iris[..., :2].reshape(batch_size, seq_len, 2)
        rotated_right_iris = rotated_right_iris[..., :2].reshape(batch_size, seq_len, 2)

        return rotated_landmarks, rotated_left_iris, rotated_right_iris

    @staticmethod
    def calculate_angle(pa, pb):
        delta = pa - pb
        delta_x = torch.where(delta[:, 0] == 0, torch.tensor(1e-8), delta[:, 0])
        angle_rad = torch.atan(delta[:, 1] / delta_x)
        angle_deg = torch.rad2deg(angle_rad)
        return torch.abs(angle_deg)

    @staticmethod
    def calculate_slope(pa, pb):
        delta = pa - pb
        slope = torch.where(torch.isclose(delta[:, 0], torch.tensor(0.0)),
                            torch.tensor(float('inf')),
                            delta[:, 1] / delta[:, 0])
        return torch.abs(slope)

    @staticmethod
    def calculate_euclidean_distance(pa, pb):
        squared_diff = (pa - pb) ** 2
        sum_squared_diff = torch.sum(squared_diff, dim=1)
        distances = torch.sqrt(sum_squared_diff)
        return distances

    @staticmethod
    def calculate_cumulative_euclidean_distance(points):
        diff = points[:, 1:] - points[:, :-1]
        distances = torch.sqrt(torch.sum(diff ** 2, dim=-1))
        total_distances = torch.sum(distances, dim=-1)
        return total_distances

    @staticmethod
    def calculate_max_ratio(a, b):
        return torch.max(a / b, b / a)

    @staticmethod
    def landmarkfeature_angle(landmarks):
        idx_pairs = [(17, 26), (19, 24), (21, 22), (31, 35), (36, 45), (48, 54), (30, 57)]
        num_samples = landmarks.shape[0]
        all_angles = []

        for i, j in idx_pairs:
            p1 = landmarks[:, i]
            p2 = landmarks[:, j]
            angles = LandmarkFeatureExtractor1.calculate_angle(p1, p2)
            all_angles.append(angles)

        all_angles = torch.stack(all_angles, dim=1)
        all_angles[:, -1] = 90 - all_angles[:, -1]

        return all_angles

    @staticmethod
    def landmarkfeature_slope(landmarks):
        idx_pairs = [(17, 26), (19, 24), (21, 22)]
        all_slopes = []

        for i, j in idx_pairs:
            p1 = landmarks[:, i]
            p2 = landmarks[:, j]
            slopes = LandmarkFeatureExtractor1.calculate_slope(p1, p2)
            all_slopes.append(slopes)

        all_slopes = torch.stack(all_slopes, dim=1)
        return all_slopes

    @staticmethod
    def landmarkfeature_max(landmarks):
        batch_size = landmarks.shape[0]
        features = []

        L = landmarks[:, 17:22, 1].mean(dim=1)
        M = landmarks[:, 22:27, 1].mean(dim=1)
        features.append(LandmarkFeatureExtractor1.calculate_max_ratio(L, M))

        Bl = torch.abs(landmarks[:, 39, 0] - landmarks[:, 36, 0])
        Br = torch.abs(landmarks[:, 45, 0] - landmarks[:, 42, 0])
        features.append(LandmarkFeatureExtractor1.calculate_max_ratio(Bl, Br))

        D = torch.abs(landmarks[:, 36, 0] - landmarks[:, 0, 0])
        E = torch.abs(landmarks[:, 16, 0] - landmarks[:, 45, 0])
        features.append(LandmarkFeatureExtractor1.calculate_max_ratio(D, E))

        H = LandmarkFeatureExtractor1.calculate_euclidean_distance(landmarks[:, 31], landmarks[:, 36])
        I = LandmarkFeatureExtractor1.calculate_euclidean_distance(landmarks[:, 35], landmarks[:, 45])
        features.append(LandmarkFeatureExtractor1.calculate_max_ratio(H, I))

        left_eye_points = landmarks[:, [37, 38, 40, 41]]
        right_eye_points = landmarks[:, [43, 44, 46, 47]]
        Nl = LandmarkFeatureExtractor1.calculate_euclidean_distance(left_eye_points[:, 0], left_eye_points[:, 3])
        Nr = LandmarkFeatureExtractor1.calculate_euclidean_distance(left_eye_points[:, 1], left_eye_points[:, 2])
        Ol = LandmarkFeatureExtractor1.calculate_euclidean_distance(right_eye_points[:, 0], right_eye_points[:, 3])
        Or = LandmarkFeatureExtractor1.calculate_euclidean_distance(right_eye_points[:, 1], right_eye_points[:, 2])
        N, O = (Nl + Nr) / 2, (Ol + Or) / 2

        # features.append(LandmarkFeatureExtractor1.calculate_max_ratio(Nl, Or))
        # features.append(LandmarkFeatureExtractor1.calculate_max_ratio(Nr, Ol))
        # features.append(LandmarkFeatureExtractor1.calculate_max_ratio(N, O))


        F = LandmarkFeatureExtractor1.calculate_euclidean_distance(landmarks[:, 36], landmarks[:, 57])
        G = LandmarkFeatureExtractor1.calculate_euclidean_distance(landmarks[:, 45], landmarks[:, 57])
        features.append(LandmarkFeatureExtractor1.calculate_max_ratio(F, G))

        upper_lip = landmarks[:, 48:52]
        lower_lip = landmarks[:, 52:56]
        outer_lip = landmarks[:, 56:60]
        Pl = LandmarkFeatureExtractor1.calculate_euclidean_distance(upper_lip[:, 1], outer_lip[:, 3])
        Pu = LandmarkFeatureExtractor1.calculate_euclidean_distance(upper_lip[:, 2], outer_lip[:, 2])
        Ql = LandmarkFeatureExtractor1.calculate_euclidean_distance(lower_lip[:, 1], lower_lip[:, 3])
        Qu = LandmarkFeatureExtractor1.calculate_euclidean_distance(lower_lip[:, 0], outer_lip[:, 0])
        Vl = LandmarkFeatureExtractor1.calculate_euclidean_distance(upper_lip[:, 0], outer_lip[:, 1])
        Vr = LandmarkFeatureExtractor1.calculate_euclidean_distance(lower_lip[:, 2], outer_lip[:, 1])
        A = torch.abs(landmarks[:, 16, 0] - landmarks[:, 0, 0])
        W = torch.abs(landmarks[:, 54, 0] - landmarks[:, 48, 0])

        Wl_points = torch.stack([upper_lip[:, 0], upper_lip[:, 1], upper_lip[:, 2], upper_lip[:, 3],
                                 outer_lip[:, 1], outer_lip[:, 2], outer_lip[:, 3], upper_lip[:, 0]], dim=1)
        Wr_points = torch.stack([upper_lip[:, 3], lower_lip[:, 0], lower_lip[:, 1], lower_lip[:, 2],
                                 lower_lip[:, 3], outer_lip[:, 0], outer_lip[:, 1], upper_lip[:, 3]], dim=1)

        Wl = LandmarkFeatureExtractor1.calculate_cumulative_euclidean_distance(Wl_points)
        Wr = LandmarkFeatureExtractor1.calculate_cumulative_euclidean_distance(Wr_points)
        features.append(LandmarkFeatureExtractor1
                        .calculate_max_ratio(Pl, Ql))

        # features.append(LandmarkFeatureExtractor1.calculate_max_ratio(Pu, Qu))
        # check_nan(LandmarkFeatureExtractor1.calculate_max_ratio(Pu, Qu), "LandmarkFeatureExtractor1.calculate_max_ratio(Pu, Qu)")
        # if torch.isinf(LandmarkFeatureExtractor1.calculate_max_ratio(Pu, Qu)).any():
        #     print("LandmarkFeatureExtractor1.calculate_max_ratio(Pu, Qu) contains Inf values")

        features.append(torch.max(Vl / A, Vr / A))
        features.append(torch.max(Pl / W, Ql / W))
        features.append(torch.max(Pu / W, Qu / W))
        features.append(torch.max(Wl / W, Wr / W))

        J = LandmarkFeatureExtractor1.calculate_euclidean_distance(landmarks[:, 31], landmarks[:, 57])
        K = LandmarkFeatureExtractor1.calculate_euclidean_distance(landmarks[:, 35], landmarks[:, 57])
        T = LandmarkFeatureExtractor1.calculate_euclidean_distance(landmarks[:, 18], landmarks[:, 57])
        U = LandmarkFeatureExtractor1.calculate_euclidean_distance(landmarks[:, 25], landmarks[:, 57])
        R = LandmarkFeatureExtractor1.calculate_euclidean_distance(landmarks[:, 20], landmarks[:, 57])
        S = LandmarkFeatureExtractor1.calculate_euclidean_distance(landmarks[:, 23], landmarks[:, 57])
        C = LandmarkFeatureExtractor1.calculate_euclidean_distance(landmarks[:, 8], landmarks[:, 57])
        X = LandmarkFeatureExtractor1.calculate_euclidean_distance(landmarks[:, 33], landmarks[:, 51])
        features.append(LandmarkFeatureExtractor1.calculate_max_ratio(J, K))
        features.append(torch.max(T / A, U / A))
        features.append(torch.max(R / A, S / A))
        features.append(C / A)
        features.append(X / A)

        return torch.stack(features, dim=1).to(landmarks.device)

    @staticmethod
    def extract_features(landmarks_tensor, left_iris_tensor, right_iris_tensor):
        landmarks_tensor, left_iris_tensor, right_iris_tensor = LandmarkFeatureExtractor1.correct_landmarks(
            landmarks_tensor, left_iris_tensor, right_iris_tensor)

        B, T = landmarks_tensor.shape[:2]
        landmarks_flat = landmarks_tensor.view(B * T, 68, 2)
        left_iris_flat = left_iris_tensor.view(B * T, 2)
        right_iris_flat = right_iris_tensor.view(B * T, 2)


        f1 = LandmarkFeatureExtractor1.landmarkfeature_angle(landmarks_flat)
        f2 = LandmarkFeatureExtractor1.landmarkfeature_slope(landmarks_flat)
        f3 = LandmarkFeatureExtractor1.landmarkfeature_max(landmarks_flat)
        feature_vector = torch.cat([f1, f2, f3], dim=-1).view(B, T, -1)

        return feature_vector
