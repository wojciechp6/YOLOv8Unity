using Assets.Scripts;
using NN;
using System;
using System.Collections.Generic;
using Unity.Barracuda;
using UnityEngine;
using UnityEngine.Profiling;
using UnityEngine.UI;

public class WebCamDetector : MonoBehaviour
{
    [Tooltip("File of YOLO model. If you want to use another than YOLOv2 tiny, it may be necessary to change some const values in YOLOHandler.cs")]
    public NNModel ModelFile;

    [Tooltip("RawImage component which will be used to draw resuls.")]
    public RawImage ImageUI;

    [Range(0.0f, 1f)]
    [Tooltip("The minimum value of box confidence below which boxes won't be drawn.")]
    public float MinBoxConfidence = 0.3f;

    protected NNHandler nn;
    protected WebCamTextureProvider CamTextureProvider;
    protected Color[] colorArray = new Color[] { Color.red, Color.green, Color.blue, Color.cyan, Color.magenta, Color.yellow };

    YOLOv8 yolo;


    private void OnEnable()
    {
        nn = new NNHandler(ModelFile);
        yolo = new YOLOv8Segmentation(nn);

        CamTextureProvider = GetTextureProvider();
    }

    private void Update()
    {
        YOLOv8OutputReader.DiscardThreshold = MinBoxConfidence;
        Texture2D texture = GetNextTexture();

        var boxes = yolo.Run(texture);
        DrawResults(boxes, texture);
        ImageUI.texture = texture;
    }

    protected WebCamTextureProvider GetTextureProvider()
    {
        var firstInput = nn.model.inputs[0];
        int height = firstInput.shape[5];
        int width = firstInput.shape[6];

        WebCamTextureProvider textureProvider = new WebCamTextureProvider(width, height);
        textureProvider.Start();
        return textureProvider;
    }

    protected Texture2D GetNextTexture()
    {
        return CamTextureProvider.GetTexture();
    }

    void OnDisable()
    {
        nn.Dispose();
        CamTextureProvider.Stop();
    }

    protected void DrawResults(IEnumerable<ResultBox> results, Texture2D img)
    {
        results.ForEach(box => DrawBox(box, img));
    }

    protected virtual void DrawBox(ResultBox box, Texture2D img)
    {
        Color boxColor = colorArray[box.bestClassIndex % colorArray.Length];
        int boxWidth = (int)(box.score / MinBoxConfidence);
        TextureTools.DrawRectOutline(img, box.rect, boxColor, boxWidth, rectIsNormalized: false, revertY: true);
    }
}
