package sample;

import javafx.collections.FXCollections;
import javafx.event.ActionEvent;
import javafx.fxml.Initializable;
import javafx.scene.control.*;
import javafx.scene.image.ImageView;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import sample.service.ImageFilterService;
import sample.service.impl.ImageFilterServiceImpl;
import sample.util.CvUtilsFX;

import java.io.File;
import java.net.URL;
import java.util.ResourceBundle;

public class Controller implements Initializable {

    public ImageView idIVSrc;
    public ImageView idIVRes;
    public TextField idTVImgPath;
    public Button idBtOpenImage;
    public Button idBtUseFilter;
    public ToggleGroup buttonGroupFilter;
    public RadioButton idRBHighPass;
    public RadioButton idRBMorph;
    public ListView idLVHighPass;
    public ListView idLVMorphOperation;
    public ListView idLVMorphMask;

    private Mat currentImage;
    private Stage stage;
    private ImageFilterService service;

    private int[] highPassFilterMaskSize = new int[]{3, 3, 3, 3, 5};
    private double[][] highPassFilterMasks = new double[][]{
            {-1, -1, -1, -1, 9, -1, -1, -1, -1},
            {1, -2, 1, -2, 5, -2, 1, -2, 1},
            {0, -1, 0, -1, 5, -1, 0, -1, 0},
            {0, -1, 0, -1, 20, -1, 0, -1, 0},
            {0, 0, -1, 0, 0, 0, -1, -2, -1, 0, -1, -2, 16, -2, -1, 0, -1, -2, -1, 0, 0, 0, -1, 0, 0,},
    };
    private String[] highPassFilterStrs = new String[]{
            "3x3:\t[-1, -1, -1]\n\t[-1,  9, -1]\n\t[-1, -1, -1]",
            "3x3:\t[ 1, -2,  1]\n\t[-2,  5, -2]\n\t[ 1, -2,  1]",
            "3x3:\t[ 0, -1,  0]\n\t[-1,  5, -1]\n\t[ 0, -1,  0]",
            "3x3:\t[ 0, -1,  0]\n\t[-1,  20,-1]\n\t[ 0, -1,  0]",
            "Laplasian of Gauss (5x5):\n\t[ 0,  0,  -1,   0,  0]\n\t[ 0, -1,  -2, -1,  0]\n\t[-1, -2, 16, -2, -1]\n\t[ 0, -1,  -2, -1,  0]\n\t[ 0,  0,  -1,   0,  0]"
    };

    private String[] morphOperationStrs = new String[]{"Erosion", "Dilation", "Opening", "Closure"};
    private int[] morphOperation = new int[]{Imgproc.MORPH_ERODE, Imgproc.MORPH_DILATE, Imgproc.MORPH_OPEN, Imgproc.MORPH_CLOSE};
    private String[] morphMaskStrs = new String[]{"3x3 square", "1x3 line", "3x1 column"};
    private double[][] morphMask = new double[][]{
            {1, 1, 1, 1, 1, 1, 1, 1, 1},
            {1, 1, 1},
            {1, 1, 1}
    };
    private int[][] morphMaskSizes = new int[][]{
            {3, 3},
            {1, 3},
            {3, 1}
    };

    @Override
    public void initialize(URL location, ResourceBundle resources) {

        service = new ImageFilterServiceImpl();

        idLVHighPass.setItems(FXCollections.observableArrayList(highPassFilterStrs));
        idLVHighPass.getSelectionModel().select(0);

        idLVMorphOperation.setItems(FXCollections.observableArrayList(morphOperationStrs));
        idLVMorphOperation.getSelectionModel().select(0);

        idLVMorphMask.setItems(FXCollections.observableArrayList(morphMaskStrs));
        idLVMorphMask.getSelectionModel().select(0);
    }

    public Stage getStage() {
        return stage;
    }

    public void setStage(Stage stage) {
        this.stage = stage;
    }

    public void chooseImage(ActionEvent actionEvent) {

        FileChooser fileChooser = new FileChooser();
        File img = fileChooser.showOpenDialog(stage);

        if (img == null) {
            return;
        }

        Mat imgMat = Imgcodecs.imread(img.getAbsolutePath());
        if (imgMat.dataAddr() == 0) {
            return;
        }

        idTVImgPath.setText(img.getAbsolutePath());
        idIVSrc.setImage(CvUtilsFX.MatToImageFX(imgMat));

        currentImage = imgMat;
        idBtUseFilter.setDisable(false);
    }

    public void filterImage(ActionEvent actionEvent) {

        Mat filteredMat = null;
        Mat mask;

        if (idRBHighPass.isSelected()) {
            int selectedItemId = idLVHighPass.getSelectionModel().getSelectedIndex();

            int maskSize = highPassFilterMaskSize[selectedItemId];
            mask = new Mat(maskSize, maskSize, CvType.CV_32FC1);
            mask.put(0, 0, highPassFilterMasks[selectedItemId]);

            filteredMat = service.highPassFilter(currentImage, mask);
        } else if (idRBMorph.isSelected()) {

            int selectedOperationId = idLVMorphOperation.getSelectionModel().getSelectedIndex();
            int selectedMaskId = idLVMorphMask.getSelectionModel().getSelectedIndex();

            int[] maskSizes = morphMaskSizes[selectedMaskId];

            mask = new Mat(maskSizes[0], maskSizes[1], CvType.CV_32FC1);
            mask.put(0, 0, morphMask[selectedMaskId]);

            filteredMat = service.morphologicalFilter(currentImage, mask, morphOperation[selectedOperationId]);
        }

        idIVRes.setImage(CvUtilsFX.MatToImageFX(filteredMat));
    }

}
