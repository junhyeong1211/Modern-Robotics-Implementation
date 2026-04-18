{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNJsjAKhlJW86x57mQqWkbn"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 13,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "QYuvqsnA1qrc",
        "outputId": "ea47ad18-7e92-4e2d-9523-3b62903a5ba0"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "손끝 위치 (지구 기준): [12.366   5.4924  0.0868]\n",
            "관절 속도: [-0.0868  0.0868  0.    ]\n",
            "조작도: 0.5000\n",
            "정상: 자유롭게 이동 가능\n"
          ]
        }
      ],
      "source": [
        "import numpy as np\n",
        "np.set_printoptions(precision=4, suppress=True)\n",
        "\n",
        "def skew(w):\n",
        "    return np.array([[ 0,   -w[2],  w[1]],\n",
        "                     [ w[2],  0,   -w[0]],\n",
        "                     [-w[1],  w[0],  0  ]])\n",
        "\n",
        "def matrix_exp_screw(S, theta):\n",
        "    w = S[:3]\n",
        "    v = S[3:]\n",
        "    if np.linalg.norm(w) < 1e-6:\n",
        "        T = np.eye(4)\n",
        "        T[:3, 3] = v * theta\n",
        "        return T\n",
        "    W = skew(w)\n",
        "    R = np.eye(3) + np.sin(theta)*W + (1-np.cos(theta))*(W @ W)\n",
        "    G = np.eye(3)*theta + (1-np.cos(theta))*W + (theta-np.sin(theta))*(W @ W)\n",
        "    T = np.eye(4)\n",
        "    T[:3, :3] = R\n",
        "    T[:3,  3] = G @ v\n",
        "    return T\n",
        "\n",
        "def adjoint(T):\n",
        "    R = T[:3, :3]\n",
        "    p = T[:3, 3]\n",
        "    Ad = np.zeros((6, 6))\n",
        "    Ad[:3, :3] = R\n",
        "    Ad[3:, :3] = skew(p) @ R\n",
        "    Ad[3:, 3:] = R\n",
        "    return Ad\n",
        "\n",
        "def manipulability(J):\n",
        "    # 평면 3R 로봇: z각속도(row 2) + x선속도(row 3) + y선속도(row 4)\n",
        "    J_planar = J[[2, 3, 4], :]\n",
        "    return np.sqrt(abs(np.linalg.det(J_planar @ J_planar.T)))\n",
        "\n",
        "# ── 시스템 설정 ──────────────────────────\n",
        "T_sb = np.array([[1,      0,       0,      10],\n",
        "                 [0,  0.9848, -0.1736,      5],\n",
        "                 [0,  0.1736,  0.9848,      0],\n",
        "                 [0,      0,       0,       1]])\n",
        "\n",
        "S1 = np.array([0, 0, 1,  0,  0, 0])\n",
        "S2 = np.array([0, 0, 1,  0, -1, 0])\n",
        "S3 = np.array([0, 0, 1,  0, -2, 0])\n",
        "\n",
        "M  = np.array([[1, 0, 0, 2.5],\n",
        "               [0, 1, 0,   0],\n",
        "               [0, 0, 1,   0],\n",
        "               [0, 0, 0,   1]])\n",
        "\n",
        "theta = [0, np.pi/6, -np.pi/6]\n",
        "\n",
        "# ── Phase 1: PoE ─────────────────────────\n",
        "T1 = matrix_exp_screw(S1, theta[0])\n",
        "T2 = matrix_exp_screw(S2, theta[1])\n",
        "T3 = matrix_exp_screw(S3, theta[2])\n",
        "T_be = T1 @ T2 @ T3 @ M\n",
        "T_se = T_sb @ T_be\n",
        "print(f\"손끝 위치 (지구 기준): {T_se[:3, 3]}\")\n",
        "\n",
        "# ── Phase 2: Jacobian ─────────────────────\n",
        "Js = np.column_stack([\n",
        "    S1,\n",
        "    adjoint(T1) @ S2,\n",
        "    adjoint(T1 @ T2) @ S3\n",
        "])\n",
        "\n",
        "V_drone_s = np.array([0, 0, 0, 0.0, -0.1736, 0.9848])\n",
        "V_body_s  = np.array([0.1, 0, 0, 0, 0, 0])\n",
        "\n",
        "# 1. 지구 기준 상대 속도\n",
        "V_rel_s = V_drone_s - V_body_s\n",
        "\n",
        "# 2. 배 기준으로 Adjoint 변환 ({s} -> {b})\n",
        "T_bs = np.linalg.inv(T_sb)\n",
        "V_desired_b = adjoint(T_bs) @ V_rel_s\n",
        "\n",
        "# 3. 관절 속도 계산\n",
        "theta_dot = np.linalg.pinv(Js) @ V_desired_b\n",
        "print(f\"관절 속도: {theta_dot}\")\n",
        "\n",
        "# ── Phase 3: 특이점 감지 ──────────────────\n",
        "w = manipulability(Js)\n",
        "print(f\"조작도: {w:.4f}\")\n",
        "\n",
        "if w < 0.1:\n",
        "    print(\"경고: 특이점 근처. 경로 재계획 필요\")\n",
        "elif w < 0.3:\n",
        "    print(\"주의: 조작도 낮음. 속도 제한 권장\")\n",
        "else:\n",
        "    print(\"정상: 자유롭게 이동 가능\")"
      ]
    }
  ]
}