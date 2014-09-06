
//Multiple face detection and recognition in real time
//Using EmguCV cross platform .Net wrapper to the Intel OpenCV image processing library for C#.Net
//Writed by Sergio Andrés Guitérrez Rojas
//"Serg3ant" for the delveloper comunity
// Sergiogut1805@hotmail.com
//Regards from Bucaramanga-Colombia ;)

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using System.IO;
using System.Diagnostics;

namespace MultiFaceRec
{

    public partial class FrmPrincipal : Form
    {
        //Declararation of all variables, vectors and haarcascades
        Image<Bgr, Byte> currentFrame;
        Capture grabber;
        HaarCascade face;
        HaarCascade eye;
        MCvFont font = new MCvFont(FONT.CV_FONT_HERSHEY_TRIPLEX, 0.5d, 0.5d);
        Image<Gray, byte> result, TrainedFace = null;
        Image<Gray, byte> gray = null;
        List<Image<Gray, byte>> trainingImages = new List<Image<Gray, byte>>();
        List<string> labels= new List<string>();
        List<string> NamePersons = new List<string>();
        int ContTrain, NumLabels, t;
        string name, names = null;


        public FrmPrincipal()
        {
            InitializeComponent();
            //Load haarcascades for face detection
            face = new HaarCascade("haarcascade_frontalface_default.xml");
            //eye = new HaarCascade("haarcascade_eye.xml");
            try
            {

                //Load of previus trainned faces and labels for each image
                string Labelsinfo = File.ReadAllText(Application.StartupPath + "/TrainedFaces/TrainedLabels.txt");
                string[] Labels = Labelsinfo.Split('%');
                NumLabels = Convert.ToInt16(Labels[0]);
                ContTrain = NumLabels;
                string LoadFaces;

                for (int tf = 1; tf < NumLabels+1; tf++)
                {
                    LoadFaces = "face" + tf + ".bmp";
                    trainingImages.Add(new Image<Gray, byte>(Application.StartupPath + "/TrainedFaces/" + LoadFaces));
                    labels.Add(Labels[tf]);
                }
            
            }
            catch(Exception e)
            {
                //MessageBox.Show(e.ToString());
                MessageBox.Show("Nothing in binary database, please add at least a face(Simply train the prototype with the Add Face Button).", "Triained faces load", MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
            }

        }



        private void button1_Click(object sender, EventArgs e)
        {
            grabber = new Capture();
            grabber.QueryFrame();
            //Initialize the FrameGraber event
            Application.Idle += new EventHandler(FrameGrabber);

         /*   OpenFileDialog Openfile = new OpenFileDialog();
            if (Openfile.ShowDialog() == DialogResult.OK)
            {
                Image<Bgr, double> My_Image = new Image<Bgr, double>(Openfile.FileName);
                Image<Gray, Byte> gray_image = My_Image.Convert<Gray, Byte>();

                imageBox1.Image = gray_image;
            }
            */
            button1.Enabled = false;
        }



        private void buttonCapture_Click(object sender, EventArgs e) 
        {

        }


        private void button2_Click(object sender, System.EventArgs e)
        {

            
            try
            {
                //Trained face counter
                ContTrain = ContTrain + 1;
                //Get a gray frame from capture device
                gray = grabber.QueryGrayFrame().Resize(320, 240, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);
                   
                //Face Detector
                MCvAvgComp[][] facesDetected = gray.DetectHaarCascade(
                face,
                2.0,
                10,
                Emgu.CV.CvEnum.HAAR_DETECTION_TYPE.DO_CANNY_PRUNING,
                new Size(200, 200));

                //Action for each element detected
                foreach (MCvAvgComp f in facesDetected[0])
                {
                    TrainedFace = currentFrame.Copy(f.rect).Convert<Gray, byte>();
                    break;
                }

                //resize face detected image for force to compare the same size with the 
                //test image with cubic interpolation type method
                TrainedFace = result.Resize(100, 100, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);
                trainingImages.Add(TrainedFace);
                labels.Add(textBox1.Text);

                //Show face added in gray scale
                imageBox1.Image = TrainedFace;

                //Write the number of triained faces in a file text for further load
                File.WriteAllText(Application.StartupPath + "/TrainedFaces/TrainedLabels.txt", trainingImages.ToArray().Length.ToString() + "%");

                //Write the labels of triained faces in a file text for further load
                for (int i = 1; i < trainingImages.ToArray().Length + 1; i++)
                {
                    trainingImages.ToArray()[i - 1].Save(Application.StartupPath + "/TrainedFaces/face" + i + ".bmp");
                    File.AppendAllText(Application.StartupPath + "/TrainedFaces/TrainedLabels.txt", labels.ToArray()[i - 1] + "%");
                }

                MessageBox.Show(textBox1.Text + "´s face detected and added :)", "Training OK", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            catch
            {
                MessageBox.Show("Enable the face detection first", "Training Fail", MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
            }
        }


        void FrameGrabber(object sender, EventArgs e)
        {
           
            Int32 frameWidth = 320;
            Int32 frameheight = 240;

            Image<Gray, Byte> lastFrame;
            Image<Gray, Byte> frameDifference = new Image<Gray, byte>(frameWidth, frameheight);
            Image<Gray, Byte> atualFrame;


            label3.Text = "0";
            //label4.Text = "";
            NamePersons.Add("");


            //Get the current frame form capture device

            lastFrame = grabber.QueryFrame().Resize(320, 240, Emgu.CV.CvEnum.INTER.CV_INTER_AREA).Convert<Gray, Byte>();

            while (lastFrame == null) {
                lastFrame = grabber.QueryFrame().Resize(320, 240, Emgu.CV.CvEnum.INTER.CV_INTER_AREA).Convert<Gray, Byte>();
            }

            currentFrame = grabber.QueryFrame().Resize(320, 240, Emgu.CV.CvEnum.INTER.CV_INTER_AREA);
            atualFrame = currentFrame.Convert<Gray, Byte>();
            CvInvoke.cvAbsDiff(lastFrame,atualFrame,frameDifference);


            Image<Gray, Byte> thresholded = new Image<Gray, byte>(frameWidth, frameheight);
            CvInvoke.cvThreshold(frameDifference, thresholded, 20, 255, THRESH.CV_THRESH_BINARY);

            Image<Gray, Byte> erosion = new Image<Gray, byte>(frameWidth, frameheight);
            CvInvoke.cvErode(thresholded, erosion, IntPtr.Zero, 2);

            drawBoxes(erosion, currentFrame);

            capturedImageBox.Image = erosion;
            imageBoxFrameGrabber.Image = currentFrame;


            //Convert it to Grayscale
                    gray = currentFrame.Convert<Gray, Byte>();

                    //Face Detector
                    MCvAvgComp[][] facesDetected = gray.DetectHaarCascade(
                  face,
                  1.2,
                  10,
                  Emgu.CV.CvEnum.HAAR_DETECTION_TYPE.DO_CANNY_PRUNING,
                  new Size(20, 20));

                    //Action for each element detected
                    foreach (MCvAvgComp f in facesDetected[0])
                    {
                        t = t + 1;
                        result = currentFrame.Copy(f.rect).Convert<Gray, byte>().Resize(100, 100, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);
                        //draw the face detected in the 0th (gray) channel with blue color
                        currentFrame.Draw(f.rect, new Bgr(Color.Red), 2);


                        if (trainingImages.ToArray().Length != 0)
                        {
                            //TermCriteria for face recognition with numbers of trained images like maxIteration
                        MCvTermCriteria termCrit = new MCvTermCriteria(ContTrain, 0.001);

                        //Eigen face recognizer
                        EigenObjectRecognizer recognizer = new EigenObjectRecognizer(
                           trainingImages.ToArray(),
                           labels.ToArray(),
                           3000,
                           ref termCrit);

                        name = recognizer.Recognize(result);

                            //Draw the label for each face detected and recognized
                        currentFrame.Draw(name, ref font, new Point(f.rect.X - 2, f.rect.Y - 2), new Bgr(Color.LightGreen));

                        }

                            NamePersons[t-1] = name;
                            NamePersons.Add("");


                        //Set the number of faces detected on the scene
                        label3.Text = facesDetected[0].Length.ToString();
                       
                        /*
                        //Set the region of interest on the faces
                        
                        gray.ROI = f.rect;
                        MCvAvgComp[][] eyesDetected = gray.DetectHaarCascade(
                           eye,
                           1.1,
                           10,
                           Emgu.CV.CvEnum.HAAR_DETECTION_TYPE.DO_CANNY_PRUNING,
                           new Size(20, 20));
                        gray.ROI = Rectangle.Empty;

                        foreach (MCvAvgComp ey in eyesDetected[0])
                        {
                            Rectangle eyeRect = ey.rect;
                            eyeRect.Offset(f.rect.X, f.rect.Y);
                            currentFrame.Draw(eyeRect, new Bgr(Color.Blue), 2);
                        }
                         */

                    }
                        t = 0;

                        //Names concatenation of persons recognized
                    for (int nnn = 0; nnn < facesDetected[0].Length; nnn++)
                    {
                        names = names + NamePersons[nnn] + ", ";
                    }
                    //Show the faces procesed and recognized
                    imageBoxFrameGrabber.Image = currentFrame;
                    label4.Text = names;
                    names = "";
                    //Clear the list(vector) of names
                    NamePersons.Clear();

                }

        private void button3_Click(object sender, EventArgs e)
        {
            Process.Start("Donate.html");
        }

        private void groupBox2_Enter(object sender, EventArgs e)
        {

        }
/*
        private void ProcessFrame(object sender, EventArgs e)
        {
            Int32 _frameWidth = 320;
            Int32 _frameHeight = 240;

            
            // Get the current frame from the camera - color and gray
            Image<Bgr, Byte> originalFrame = grabber.QueryFrame();

            // This usually occurs when using a video file - after the last frame is read
            // the next frame is null
            while (originalFrame == null)
            {
                // Reset the camera since no frame was captured - for videos, restart the video playback
                originalFrame = grabber.QueryFrame();
            }

            Image<Bgr, Byte> image = originalFrame.Resize(_frameWidth, _frameHeight, 0);
            Image<Gray, Byte> frame = image.Convert<Gray, Byte>();

            // Perform differencing on them to find the "new introductions to the background" and "motions"
            Image<Gray, Byte> BgDifference = new Image<Gray, byte>(_frameWidth, _frameHeight);
            Image<Gray, Byte> FrameDifference = new Image<Gray, byte>(_frameWidth, _frameHeight);
            CvInvoke.cvAbsDiff(grabber.QueryFrame().Resize(), frame, BgDifference);
            CvInvoke.cvAbsDiff((_lastFrame == null) ? frame : _lastFrame, frame, FrameDifference);

            // Perform thresholding to remove noise and boost "new introductions"
            Image<Gray, Byte> thresholded = new Image<Gray, byte>(_frameWidth, _frameHeight);
            CvInvoke.cvThreshold(BgDifference, thresholded, 20, 255, THRESH.CV_THRESH_BINARY);

            // Perform erosion to remove camera noise
            Image<Gray, Byte> eroded = new Image<Gray, byte>(_frameWidth, _frameHeight);
            CvInvoke.cvErode(thresholded, eroded, IntPtr.Zero, 2);

            // Takes the thresholded image and looks for squares and draws the squares out on top of the current frame
            drawBoxes(eroded, image);

            // Put the captured frame in the imagebox
            capturedImageBox.Image = image;
            // Store the current frame in the _lastFrame variable - it becomes the last frame now
            _lastFrame = image.Convert<Gray, Byte>();

            // Draw the frame-to-frame difference (motion) on to the imgImageBox image box
            imgImageBox.Image = FrameDifference;

            // Draw the thresholded image in the motionImageBox image box - so that we can view it
            motionImageBox.Image = eroded;

            // Move the background close to the current frame
            if (_adaptiveBackground == true)
            {
                Image<Gray, Byte> newBackground = new Image<Gray, byte>(_frameWidth, _frameHeight);
                MoveToward(ref _backgroundImage, ref frame, ref newBackground, _backgroundAdaptionRate);
                _backgroundImage = newBackground;
            }
            grayImageBox.Image = _backgroundImage;
        }*/
        private void drawBoxes(Emgu.CV.Image<Gray, Byte> img, Emgu.CV.Image<Bgr, Byte> original)
        {

            Gray cannyThreshold = new Gray(180);
            Gray cannyThresholdLinking = new Gray(120);
            Gray circleAccumulatorThreshold = new Gray(120);

            Image<Gray, Byte> cannyEdges = img.Canny(cannyThreshold, cannyThresholdLinking);
            LineSegment2D[] lines = cannyEdges.HoughLinesBinary(
                2, //Distance resolution in pixel-related units
                Math.PI / 45.0, //Angle resolution measured in radians.
                20, //threshold
                30, //min Line width
                10 //gap between lines
                )[0]; //Get the lines from the first channel


            #region Find rectangles
            List<MCvBox2D> boxList = new List<MCvBox2D>();

            using (MemStorage storage = new MemStorage()) //allocate storage for contour approximation
                for (Contour<Point> contours = cannyEdges.FindContours(); contours != null; contours = contours.HNext)
                {
                    Contour<Point> currentContour = contours.ApproxPoly(contours.Perimeter * 0.05, storage);

                    if (contours.Area > 250) //only consider contours with area greater than 250
                    {
                        if (currentContour.Total == 4) //The contour has 4 vertices.
                        {
                            #region determine if all the angles in the contour are within the range of [80, 100] degree
                            bool isRectangle = true;
                            Point[] pts = currentContour.ToArray();
                            LineSegment2D[] edges = PointCollection.PolyLine(pts, true);

                            for (int i = 0; i < edges.Length; i++)
                            {
                                double angle = Math.Abs(
                                   edges[(i + 1) % edges.Length].GetExteriorAngleDegree(edges[i]));
                                if (angle < 80 || angle > 100)
                                {
                                    isRectangle = false;
                                    break;
                                }
                            }
                            #endregion

                            if (isRectangle) boxList.Add(currentContour.GetMinAreaRect());
                        }
                    }
                }
            #endregion

            #region draw rectangles
            Image<Bgr, Byte> rectangleImage = new Image<Bgr, byte>(img.Width, img.Height);
            foreach (MCvBox2D box in boxList)
            {
                rectangleImage.Draw(box, new Bgr(Color.DarkOrange), 2);
                original.Draw(box, new Bgr(Color.DarkOrange), 2);
            }

            capturedImageBox.Image = rectangleImage;
            #endregion
        }

        
    }
 
}