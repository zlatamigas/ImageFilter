package sample;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.image.Image;
import javafx.stage.Stage;
import org.opencv.core.Core;

public class Main extends Application {

    private static String APP_NAME = "Image filter";
    private static String APP_PAGE = "sample.fxml";
    private static String STYLESHEET = "application.css";
    private static String ICON = "icon.png";

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    @Override
    public void start(Stage primaryStage) throws Exception {

        FXMLLoader loader = new FXMLLoader(getClass().getResource(APP_PAGE));
        Parent root = (Parent) loader.load();

        Scene scene = new Scene(root);
        scene.getStylesheets().add(getClass().getResource(STYLESHEET).toExternalForm());

        Controller controller = (Controller) loader.getController();
        controller.setStage(primaryStage);

        primaryStage.getIcons().add(new Image(Main.class.getResourceAsStream(ICON)));
        primaryStage.setTitle(APP_NAME);
        primaryStage.setScene(scene);
        primaryStage.show();
    }


    public static void main(String[] args) {

        launch(args);
    }
}
