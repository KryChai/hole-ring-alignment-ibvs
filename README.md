# Hole-Ring Alignment IBVS

Image-based visual servoing to align a picked workpiece with a base carrying a corresponding circular hole pattern. The center and yaw must both agree before placement. A difference in the apparent radii can make direct pixel-to-pixel matching of the holes misleading.

## Method

1. Fit a circle to the visible holes and order them around the fitted center. Freeze the held workpiece's image reference when starting placement alignment.
2. Form an error from the two center coordinates and the wrapped angular difference of each corresponding radial direction. Six-point mode uses two directions per ring. Four-point mode uses one. Neither mode requires equal apparent radii.
3. Choose a cyclic correspondence from the initial angles and track it across observations. Reject a correspondence jump if it is too large relative to adjacent hole spacing. A symmetric ring identifies a local alignment branch, so unique absolute indexing needs an additional visual landmark.
4. Estimate an image Jacobian from small X, Y and yaw probes. Normalize the position and angle errors, solve a damped least-squares step with bounded motion, and update the Jacobian from actual motion and fresh observations. The demo requires consecutive in-tolerance observations.

The pickup variant fits the complete hole ring directly in image pixels and drives its center to the image center. The `online_learn_J` variant updates the initial XY image Jacobian using a bounded rank-one Broyden correction. The method requires an initial response with plausible axis directions. Physical placement depth, fixture geometry and hardware operating limits are supplied separately by an integration.

## Reproduce with synthetic data

```sh
python -m pip install -r requirements.txt
python demo.py
python demo.py --references 1
python -m pytest -q
```

`alignment.py` contains the pixel geometry, correspondence, Jacobian estimation and controller. `pickup.py` contains the pixel-ring pickup on branches that include it. The example reports pixel and yaw convergence with two different projected radii. The numbers are simulation results.

## Branches

This snapshot is based on `master`. Original four- and six-point placement algorithms.

- [`master`](https://github.com/KryChai/hole-ring-alignment-ibvs/tree/master) — four- and six-point placement baseline.
- [`dev`](https://github.com/KryChai/hole-ring-alignment-ibvs/tree/dev) — fixed-Jacobian pixel pickup and placement.
- [`online_learn_J`](https://github.com/KryChai/hole-ring-alignment-ibvs/tree/online_learn_J) — adds bounded online XY Jacobian learning during pickup.

## 中文说明

项目解决环形孔阵工件在放下前如何同时对中、对角的问题。六点方法从工件和底座各取一个圆心及两个参考孔，用圆心差纠正 XY，用径向角差纠正 yaw。两圈的成像半径可以不同。控制器通过小幅探测估计图像雅可比，再用实际位移和新图像更新。仿真可复现四点、六点闭环及孔位跨帧关联。在线学习分支还提供像素取件的有界雅可比更新。
