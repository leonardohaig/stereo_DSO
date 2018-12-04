/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


#pragma once

 
#include "util/NumType.h"
#include "vector"
#include <math.h>
#include "util/settings.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "IOWrapper/Output3DWrapper.h"




namespace dso
{
struct CalibHessian;
struct FrameHessian;
struct PointFrameResidual;

class CoarseTracker {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	CoarseTracker(int w, int h);
	~CoarseTracker();

	bool trackNewestCoarse(
			FrameHessian* newFrameHessian,
			SE3 &lastToNew_out, AffLight &aff_g2l_out,
			int coarsestLvl, Vec5 minResForAbort,
			IOWrap::Output3DWrapper* wrap=0);

	//求图像金字塔各层的权重等变量，貌似只初始化一次即可。
	void setCTRefForFirstFrame(
			std::vector<FrameHessian*> frameHessians);

	void setCoarseTrackingRef(
			std::vector<FrameHessian*> frameHessians, FrameHessian* fh_right, CalibHessian Hcalib);
	
	void makeCoarseDepthForFirstFrame(FrameHessian* fh);

	//求图像金字塔各层的内参矩阵K，及Ki，w，h等参数，可以理解为该函数只运行一次
	void makeK(
			CalibHessian* HCalib);

	bool debugPrint, debugPlot;

	//图像金字塔？可以这样理解吗?[0]表示输入的图像（可能裁剪过）的信息，如：焦距，像主点，图像大小
	Mat33f K[PYR_LEVELS];//每层图像的内参矩阵。3X3大小。
	Mat33f Ki[PYR_LEVELS];//每层图像的内参矩阵的逆矩阵。i--inverse
	float fx[PYR_LEVELS];//每层图像的fx
	float fy[PYR_LEVELS];//每层图像的fy
	float fxi[PYR_LEVELS];//对应Ki的[0][0],可以理解为fx的逆吧
	float fyi[PYR_LEVELS];//对应Ki的[1][1]，可以理解为fy的逆吧
	float cx[PYR_LEVELS];//每层图像的cx
	float cy[PYR_LEVELS];//每层图像的cy
	float cxi[PYR_LEVELS];//对应Ki的[0][2]，可以理解为cx的逆吧
	float cyi[PYR_LEVELS];//对应Ki的[1][2]，可以理解为cy的逆吧
	int w[PYR_LEVELS];//w,h,[0]表示原始图像大小，设定的图像大小，每层图像的宽带
	int h[PYR_LEVELS];//每层图像的高度

    void debugPlotIDepthMap(float* minID, float* maxID, std::vector<IOWrap::Output3DWrapper*> &wraps);
    void debugPlotIDepthMapFloat(std::vector<IOWrap::Output3DWrapper*> &wraps);


	FrameHessian* lastRef;
	AffLight lastRef_aff_g2l;
	FrameHessian* newFrame;
	int refFrameID;

	// act as pure ouptut
	Vec5 lastResiduals;
	Vec3 lastFlowIndicators;
	double firstCoarseRMSE;
private:

	//求图像金字塔对应的变量，该函数求出的变量内容为函数声明下方的三个float* 变量
	void makeCoarseDepthL0(std::vector<FrameHessian*> frameHessians, FrameHessian* fh_right, CalibHessian Hcalib);
	float* idepth[PYR_LEVELS];//每层图像对应的逆深度。嗯，这样理解应该没问题
	float* weightSums[PYR_LEVELS];//每层图像对应的“权重”，嗯，应该可以这样理解
	float* weightSums_bak[PYR_LEVELS];//类似于上面2个变量，只是不清楚该变量叫什么名字合适。。。


	Vec6 calcResAndGS(int lvl, Mat88 &H_out, Vec8 &b_out, SE3 refToNew, AffLight aff_g2l, float cutoffTH);
	//初步计算投影误差,为SSE计算求相关变量(warped buffers)
	//返回值:0误差值 1参与误差计算的像素个数 2相机平移后带来的位置变化3常值0;4相机旋转平移后带来的位置变化 5超出阈值的比例
	Vec6 calcRes(int lvl, SE3 refToNew, AffLight aff_g2l, float cutoffTH);//计算残差
	void calcGSSSE(int lvl, Mat88 &H_out, Vec8 &b_out, SE3 refToNew, AffLight aff_g2l);//计算残差的线性化.线性化是优化的前提，在程序中，对于每个点Hessian的计算有三种方式，分别是激活点，线性化点和边缘化点，
	void calcGS(int lvl, Mat88 &H_out, Vec8 &b_out, SE3 refToNew, AffLight aff_g2l);

	// pc buffers
	float* pc_u[PYR_LEVELS];
	float* pc_v[PYR_LEVELS];
	float* pc_idepth[PYR_LEVELS];
	float* pc_color[PYR_LEVELS];
	int pc_n[PYR_LEVELS];

	// warped buffers
	float* buf_warped_idepth;
	float* buf_warped_u;
	float* buf_warped_v;
	float* buf_warped_dx;
	float* buf_warped_dy;
	float* buf_warped_residual;
	float* buf_warped_weight;
	float* buf_warped_refColor;
	int buf_warped_n;

	Accumulator9 acc;
};


class CoarseDistanceMap {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	CoarseDistanceMap(int w, int h);
	~CoarseDistanceMap();

	void makeDistanceMap(
			std::vector<FrameHessian*> frameHessians,
			FrameHessian* frame);

	void makeInlierVotes(
			std::vector<FrameHessian*> frameHessians);

	void makeK( CalibHessian* HCalib);


	float* fwdWarpedIDDistFinal;

	Mat33f K[PYR_LEVELS];
	Mat33f Ki[PYR_LEVELS];
	float fx[PYR_LEVELS];
	float fy[PYR_LEVELS];
	float fxi[PYR_LEVELS];
	float fyi[PYR_LEVELS];
	float cx[PYR_LEVELS];
	float cy[PYR_LEVELS];
	float cxi[PYR_LEVELS];
	float cyi[PYR_LEVELS];
	int w[PYR_LEVELS];
	int h[PYR_LEVELS];

	void addIntoDistFinal(int u, int v);


private:

	PointFrameResidual** coarseProjectionGrid;
	int* coarseProjectionGridNum;
	Eigen::Vector2i* bfsList1;
	Eigen::Vector2i* bfsList2;

	void growDistBFS(int bfsNum);
};

}

