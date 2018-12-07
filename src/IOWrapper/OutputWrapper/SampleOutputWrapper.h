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

//来源于github的代码参考信息
/* ======================= Some typical usecases: ===============
 *
 * (1) always get the pose of the most recent frame:
 *     -> Implement [publishCamPose].
 *
 * (2) always get the depthmap of the most recent keyframe
 *     -> Implement [pushDepthImageFloat] (use inverse depth in [image], and pose / frame information from [KF]).
 *
 * (3) accumulate final model
 *     -> Implement [publishKeyframes] (skip for final!=false), and accumulate frames.
 *
 * (4) get evolving model in real-time
 *     -> Implement [publishKeyframes] (update all frames for final==false).
 *
 *
 *
 *
 * ==================== How to use the structs: ===================
 * [FrameShell]: minimal struct kept for each frame ever tracked.
 *      ->camToWorld = camera to world transformation
 *      ->poseValid = false if [camToWorld] is invalid (only happens for frames during initialization).
 *      ->trackingRef = Shell of the frame this frame was tracked on.
 *      ->id = ID of that frame, starting with 0 for the very first frame.
 *
 *      ->incoming_id = ID passed into [addActiveFrame( ImageAndExposure* image, int id )].
 *	->timestamp = timestamp passed into [addActiveFrame( ImageAndExposure* image, int id )] as image.timestamp.
 *
 * [FrameHessian]
 *      ->immaturePoints: contains points that have not been "activated" (they do however have a depth initialization).
 *      ->pointHessians: contains active points.
 *      ->pointHessiansMarginalized: contains marginalized points.
 *      ->pointHessiansOut: contains outlier points.
 *
 *      ->frameID: incremental ID for keyframes only.
 *      ->shell: corresponding [FrameShell] struct.
 *
 *
 * [CalibHessian]
 *      ->fxl(), fyl(), cxl(), cyl(): get optimized, most recent (pinhole) camera intrinsics.
 *
 *
 * [PointHessian]
 * 	->u,v: pixel-coordinates of point.
 *      ->idepth_scaled: inverse depth of point.
 *                       DO NOT USE [idepth], since it may be scaled with [SCALE_IDEPTH] ... however that is currently set to 1 so never mind.
 *      ->host: pointer to host-frame of point.
 *      ->status: current status of point (ACTIVE=0, INACTIVE, OUTLIER, OOB, MARGINALIZED)
 *      ->numGoodResiduals: number of non-outlier residuals supporting this point (approximate).
 *      ->maxRelBaseline: value roughly proportional to the relative baseline this point was observed with (0 = no baseline).
 *                        points for which this value is low are badly contrained.
 *      ->idepth_hessian: hessian value (inverse variance) of inverse depth.
 *
 * [ImmaturePoint]
 * 	->u,v: pixel-coordinates of point.
 *      ->idepth_min, idepth_max: the initialization sais that the inverse depth of this point is very likely
 *        between these two thresholds (their mean being the best guess)
 *      ->host: pointer to host-frame of point.
 */



#pragma once
#include "util/MinimalImage.h"
#include "IOWrapper/Output3DWrapper.h"



#include "FullSystem/HessianBlocks.h"
#include "util/FrameShell.h"

namespace dso
{

class FrameHessian;
class CalibHessian;
class FrameShell;


namespace IOWrap
{

class SampleOutputWrapper : public Output3DWrapper
{
public:
        inline SampleOutputWrapper()
        {
            printf("OUT: Created SampleOutputWrapper\n");
        }

        virtual ~SampleOutputWrapper()
        {
            printf("OUT: Destroyed SampleOutputWrapper\n");
        }

        virtual void publishGraph(const std::map<long,Eigen::Vector2i> &connectivity)
        {
            printf("OUT: got graph with %d edges\n", (int)connectivity.size());

            int maxWrite = 5;

            for(const std::pair<long,Eigen::Vector2i> &p : connectivity)
            {
                int idHost = p.first>>32;
                int idTarget = p.first & 0xFFFFFFFF;
                printf("OUT: Example Edge %d -> %d has %d active and %d marg residuals\n", idHost, idTarget, p.second[0], p.second[1]);
                maxWrite--;
                if(maxWrite==0) break;
            }
        }



        virtual void publishKeyframes( std::vector<FrameHessian*> &frames, bool final, CalibHessian* HCalib)
        {
            for(FrameHessian* f : frames)
            {
                printf("OUT: KF %d (%s) (id %d, tme %f): %d active, %d marginalized, %d immature points. CameraToWorld:\n",
                       f->frameID,
                       final ? "final" : "non-final",
                       f->shell->incoming_id,
                       f->shell->timestamp,
                       (int)f->pointHessians.size(), (int)f->pointHessiansMarginalized.size(), (int)f->immaturePoints.size());
                std::cout << f->shell->camToWorld.matrix3x4() << "\n";


                int maxWrite = 5;
                for(PointHessian* p : f->pointHessians)
                {
                    printf("OUT: Example Point x=%.1f, y=%.1f, idepth=%f, idepth std.dev. %f, %d inlier-residuals\n",
                           p->u, p->v, p->idepth_scaled, sqrt(1.0f / p->idepth_hessian), p->numGoodResiduals );
                    maxWrite--;
                    if(maxWrite==0) break;
                }
            }
#if 0
            //========================保存为点云格式  start=========================//
            //TODO：暂时添加该部分，后续考虑如何用。
            float fx = HCalib->fxl();
            float fy = HCalib->fyl();
            float cx = HCalib->cxl();
            float cy = HCalib->cyl();
            float fxi = 1/fx;
            float fyi = 1/fy;
            float cxi = -cx / fx;
            float cyi = -cy / fy;
            // Open stream to write in file "points.ply"
            std::ofstream output_points;
            output_points.open("points.ply", std::ios_base::app);
            for(FrameHessian* f : frames)
            {
                auto const & m =  f->shell->camToWorld.matrix3x4();
                auto const & points = f->pointHessiansMarginalized;
                for (auto const * p : points)
                {
                    float depth = 1.0f / p->idepth;
                    auto const x = (p->u * fxi + cxi) * depth;
                    auto const y = (p->v * fyi + cyi) * depth;
                    auto const z = depth * (1 + 2*fxi);
                    Eigen::Vector4d camPoint(x, y, z, 1.f);
                    Eigen::Vector3d worldPoint = m * camPoint;
                    output_points << worldPoint.transpose() << std::endl;
                }
            }

            // Close steam
            output_points.close();

            //========================保存为点云格式  end=========================//
#endif

        }

        virtual void publishCamPose(FrameShell* frame, CalibHessian* HCalib)
        {
            printf("OUT: Current Frame %d (time %f, internal ID %d). CameraToWorld:\n",
                   frame->incoming_id,
                   frame->timestamp,
                   frame->id);
            std::cout << frame->camToWorld.matrix3x4() << "\n";

#if 0
            //======================计算相机在两帧之间的运动距离，旋转角度 start=====================//
            //TODO：暂时添加该部分，后续考虑如何用。
            // after two images input, previous and current camera poses, named preCamPose, curCamPose are available.
            //Then, I calculate transformation between the previous and current frame.
            SE3 SE3_pre  = frame->camToWorld;
            SE3 SE3_cur = frame->camToWorld;
            SE3 Transform = SE3_cur*SE3_pre.inverse();
            //According to Transform, rotation angle and translation distance can be calculated.
            Eigen::Vector3d translation =  Transform.translation();
            Eigen::Matrix3d rotation = Transform.rotationMatrix();

            double rotate_x = atan2((double)rotation(2,1) , (double)rotation(2,2));
            double rotate_y = atan2((double)-rotation(2,0), (double)sqrt((double)(rotation(2,1)*rotation(2,1) +rotation(2,2)*rotation(2,2))));
            double rotate_z = atan2((double)rotation(1,0), (double)rotation(0,0));
            double trans_x = translation(0);
            double trans_y = translation(1);
            double trans_z = translation(2);
            //======================计算相机在两帧之间的运动距离，旋转角度 end=====================//
#endif
        }


        virtual void pushLiveFrame(FrameHessian* image)
        {
            // can be used to get the raw image / intensity pyramid.
        }

        virtual void pushDepthImage(MinimalImageB3* image)
        {
            // can be used to get the raw image with depth overlay.
        }
        virtual bool needPushDepthImage()
        {
            return false;
        }

        virtual void pushDepthImageFloat(MinimalImageF* image, FrameHessian* KF )
        {
            printf("OUT: Predicted depth for KF %d (id %d, time %f, internal frame-ID %d). CameraToWorld:\n",
                   KF->frameID,
                   KF->shell->incoming_id,
                   KF->shell->timestamp,
                   KF->shell->id);
            std::cout << KF->shell->camToWorld.matrix3x4() << "\n";

            int maxWrite = 5;
            for(int y=0;y<image->h;y++)
            {
                for(int x=0;x<image->w;x++)
                {
                    if(image->at(x,y) <= 0) continue;

                    printf("OUT: Example Idepth at pixel (%d,%d): %f.\n", x,y,image->at(x,y));
                    maxWrite--;
                    if(maxWrite==0) break;
                }
                if(maxWrite==0) break;
            }
        }


};



}



}
