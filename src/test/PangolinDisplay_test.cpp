//
// Created by liheng on 12/5/18.
//
#include <iostream>
#include <pangolin/pangolin.h>
#include <Eigen/src/Core/Matrix.h>
#include <util/DatasetReader.h>

struct CustomType {
    CustomType() : x(0), y(0.0f) {}

    CustomType(int x, float y, std::string z) : x(x), y(y), z(z) {}

    int x;
    float y;
    std::string z;
};

std::ostream &operator<<(std::ostream &os, const CustomType &o) {
    os << o.x << " " << o.y << " " << o.z;
    return os;
}

std::istream &operator>>(std::istream &is, CustomType &o) {
    is >> o.x;
    is >> o.y;
    is >> o.z;
    return is;
}

void SampleMethod() {
    std::cout << "You typed ctrl-r or pushed reset" << std::endl;
}

int main(/*int argc, char* argv[]*/) {
    // Load configuration data
    pangolin::ParseVarsFile("app.cfg");

    // Create OpenGL window in single line
    pangolin::CreateWindowAndBind("Main", 640, 480);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 1000),
            pangolin::ModelViewLookAt(-0, 0.5, -3, 0, 0, 0, pangolin::AxisY));

    const int UI_WIDTH = 180;

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View &d_cam =
            pangolin::CreateDisplay()
                    .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0,
                               -640.0f / 480.0f)
                    .SetHandler(new pangolin::Handler3D(s_cam));

    // Add named Panel and bind to variables beginning 'ui'
    // A Panel is just a View with a default layout and input handling
    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0,
                                          pangolin::Attach::Pix(UI_WIDTH));

    // Safe and efficient binding of named variables.
    // Specialisations mean no conversions take place for exact types
    // and conversions between scalar types are cheap.
    pangolin::Var<bool> a_button("ui.A_Button", false, false);
    pangolin::Var<double> a_double("ui.A_Double", 3, 0, 5);
    pangolin::Var<int> an_int("ui.An_Int", 2, 0, 5);
    pangolin::Var<double> a_double_log("ui.Log_scale var", 3, 1, 1E4, true);
    pangolin::Var<bool> a_checkbox("ui.A_Checkbox", false, true);
    pangolin::Var<int> an_int_no_input("ui.An_Int_No_Input", 2);
    pangolin::Var<CustomType> any_type("ui.Some_Type",
                                       CustomType(0, 1.2f, "Hello"));

    pangolin::Var<bool> save_window("ui.Save_Window", false, false);
    pangolin::Var<bool> save_cube("ui.Save_Cube", false, false);

    pangolin::Var<bool> record_cube("ui.Record_Cube", false, false);

    const size_t num_pts = 1000;
    float somedata[3 * num_pts];
    // fill in somedata with contiguous 3 vectors representing 3D points
    // ...

    // Create pangolin::GlBuffer object encapsulating
    // GL buffer object (glBufferData/glDeleteBuffers)
    pangolin::GlBuffer glxyz(pangolin::GlArrayBuffer, num_pts, GL_FLOAT, 3,
                             GL_STATIC_DRAW);

    // Upload host data to OpenGL Buffer
    glxyz.Upload(somedata, 3 * sizeof(float) * num_pts);

    // ....
    // Actually render the points, assuming model-view and
    // projection matrices have been setup
    pangolin::RenderVbo(glxyz, GL_POINTS);

    // std::function objects can be used for Var's too. These work great with
    // C++11 closures.
    pangolin::Var<std::function<void(void)>> reset("ui.Reset", SampleMethod);

    // Demonstration of how we can register a keyboard hook to alter a Var
    pangolin::RegisterKeyPressCallback(
            pangolin::PANGO_CTRL + 'b',
            pangolin::SetVarFunctor<double>("ui.A Double", 3.5));

    // Demonstration of how we can register a keyboard hook to trigger a method
    pangolin::RegisterKeyPressCallback(pangolin::PANGO_CTRL + 'r', SampleMethod);





    float cx=312.905487;
    float fx = 370.505432;
    float cy = 918.396581;
    float fy = 918.396851;

    // show ground truth
    std::string gtPath = "/home/liheng/CLionProjects/StereoSlamData/data_odometry/00/00.txt";
    std::ifstream ReadFile(gtPath.c_str());
    std::string temp;
    std::string delim (" ");
    std::vector<std::string> results;
    Sophus::Matrix4f gtCam;
    std::vector<Sophus::Matrix4f> matrix_result;

    while(std::getline(ReadFile, temp))
    {
        split(temp, delim, results);
        for(int i = 0; i < 3; i++)
        {
            for(int j = 0; j < 4; j++)
            {
                gtCam(i,j) = atof(results[4*i + j].c_str());
            }
        }

        gtCam(3,0) = 0;
        gtCam(3,1) = 0;
        gtCam(3,2) = 0;
        gtCam(3,3) = 1;

        results.clear();
        matrix_result.push_back(gtCam);

    }

    float yellow[3] = {1,1,0};











    // Default hooks for exiting (Esc) and fullscreen (tab).
    while (!pangolin::ShouldQuit())
    {
        // Clear entire screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (pangolin::Pushed(a_button))
            std::cout << "You Pushed a button!" << std::endl;

        // Overloading of Var<T> operators allows us to treat them like
        // their wrapped types, eg:
        if (a_checkbox)
            an_int = (int)a_double;

        if (!any_type->z.compare("robot"))
            any_type = CustomType(1, 2.3f, "Boogie");

        an_int_no_input = an_int;

        if (pangolin::Pushed(save_window))
            pangolin::SaveWindowOnRender("window");

        if (pangolin::Pushed(save_cube))
            d_cam.SaveOnRender("cube");

        if (pangolin::Pushed(record_cube))
            pangolin::DisplayBase().RecordOnRender(
                    "ffmpeg:[fps=50,bps=8388608,unique_filename]//screencap.avi");

        // Activate efficiently by object
        d_cam.Activate(s_cam);

        // Render some stuff
        glColor3f(1.0, 1.0, 1.0);
        pangolin::glDrawColouredCube();

        for(unsigned int i=0;i<matrix_result.size();i++)
        {
            float sz=0.2;

            glPushMatrix();

            glMultMatrixf((GLfloat*)matrix_result[i].data());

            if(yellow == 0)
            {
                glColor3f(1,0,0);
            }
            else
                glColor3f(yellow[0],yellow[1],yellow[2]);

            glLineWidth(5);
            glBegin(GL_LINES);
            glVertex3f(0,0,0);
            glVertex3f(sz*(0-cx)/fx,sz*(0-cy)/fy,sz);

            glEnd();
            glPopMatrix();
        }







        // Swap frames and Process Events
        pangolin::FinishFrame();
    }

    return 0;
}
