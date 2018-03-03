
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
using Emgu.CV.VideoSurveillance;


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
        private MotionHistory _motionHistory;
        private IBGFGDetector<Bgr> _forgroundDetector;

        public FrmPrincipal()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            grabber = new Capture();
            grabber.QueryFrame();
            //Initialize the FrameGraber event
            Application.Idle += new EventHandler(FrameGrabber);
            button1.Enabled = false;
        }

        void FrameGrabber(object sender, EventArgs e)
        {
           
            Int32 frameWidth = 320;
            Int32 frameheight = 240;

            Image<Gray, Byte> lastFrame;
            Image<Gray, Byte> frameDifference = new Image<Gray, byte>(frameWidth, frameheight);
            Image<Gray, Byte> atualFrame;

            _motionHistory = new MotionHistory(
                1.0, //in second, the duration of motion history you wants to keep
                0.05, //in second, maxDelta for cvCalcMotionGradient
                0.5); //in second, minDelta for cvCalcMotionGradient

            //Get the current frame form capture device

            lastFrame = grabber.QueryFrame().Resize(frameWidth, frameheight, Emgu.CV.CvEnum.INTER.CV_INTER_AREA).Convert<Gray, Byte>();

            while (lastFrame == null) {
                lastFrame = grabber.QueryFrame().Resize(frameWidth, frameheight, Emgu.CV.CvEnum.INTER.CV_INTER_AREA).Convert<Gray, Byte>();
            }

            currentFrame = grabber.QueryFrame().Resize(frameWidth, frameheight, Emgu.CV.CvEnum.INTER.CV_INTER_AREA);
            

            atualFrame = currentFrame.Convert<Gray, Byte>();
            CvInvoke.cvAbsDiff(lastFrame,atualFrame,frameDifference);
            
            Image<Gray, Byte> thresholded = new Image<Gray, byte>(frameWidth, frameheight);
            CvInvoke.cvThreshold(frameDifference, thresholded, 20, 255, THRESH.CV_THRESH_BINARY);

            Image<Gray, Byte> erosion = new Image<Gray, byte>(frameWidth, frameheight);
            CvInvoke.cvErode(thresholded, erosion, IntPtr.Zero, 2);

            drawBoxes(thresholded, currentFrame);
 
      }

        private void drawBoxes(Emgu.CV.Image<Gray, Byte> img, Emgu.CV.Image<Bgr, Byte> original)
        {

            Gray cannyThreshold = new Gray(180);
            Gray cannyThresholdLinking = new Gray(120);
            Gray circleAccumulatorThreshold = new Gray(120);

            if (_forgroundDetector == null)
            {
                //_forgroundDetector = new BGCodeBookModel<Bgr>();
                _forgroundDetector = new FGDetector<Bgr>(Emgu.CV.CvEnum.FORGROUND_DETECTOR_TYPE.FGD);
                //_forgroundDetector = new BGStatModel<Bgr>(image, Emgu.CV.CvEnum.BG_STAT_TYPE.FGD_STAT_MODEL);
            }

            Image<Gray, Byte> cannyEdges = img.Canny(cannyThreshold, cannyThresholdLinking);
            _forgroundDetector.Update(original);
            _motionHistory.Update(img);

            #region get a copy of the motion mask and enhance its color
            double[] minValues, maxValues;
            Point[] minLoc, maxLoc;
            _motionHistory.Mask.MinMax(out minValues, out maxValues, out minLoc, out maxLoc);
            Image<Gray, Byte> motionMask = _motionHistory.Mask.Mul(255.0 / maxValues[0]);
            #endregion


            LineSegment2D[] lines = cannyEdges.HoughLinesBinary(
                2, //Distance resolution in pixel-related units
                Math.PI / 45.0, //Angle resolution measured in radians.
                20, //threshold
                30, //min Line width
                10 //gap between lines
                )[0]; //Get the lines from the first channel
            Int32 a = lines.Length;


            #region Find rectangles
            List<MCvBox2D> boxList = new List<MCvBox2D>();

            using (MemStorage storage = new MemStorage()) //allocate storage for contour approximation
                for (Contour<Point> contours = motionMask.FindContours(); contours != null; contours = contours.HNext)
                {
                    Contour<Point> currentContour = contours.ApproxPoly(contours.Perimeter * 0.05, storage);
                    
                    if (contours.Area > 100) //only consider contours with area greater than 250
                    {
                        if (currentContour.Total >= 4) //The contour has 4 vertices.
                        {
                            #region determine if all the angles in the contour are within the range of [90, 100] degree
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

                            if (!isRectangle) boxList.Add(currentContour.GetMinAreaRect());
                             
                        }
                    }
                     
 
                }
            #endregion

            #region draw rectangles
            Image<Bgr, Byte> rectangleImage = new Image<Bgr, byte>(img.Width, img.Height);

            foreach (MCvBox2D box in boxList)
            {
                rectangleImage.Draw(box, new Bgr(Color.Red), 2);
                original.Draw(box, new Bgr(Color.Red), 2);
            }

            capturedImageBox.Image = motionMask;
            imageBoxFrameGrabber.Image = original;
            #endregion
        }
    }
}