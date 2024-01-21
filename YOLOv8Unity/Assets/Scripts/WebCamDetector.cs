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
    [Tooltip("Text file with classes names separated by coma ','")]
    public TextAsset ClassesFile;

    [Tooltip("RawImage component which will be used to draw resuls.")]
    public RawImage ImageUI;

    [Range(0.0f, 1f)]
    [Tooltip("The minimum value of box confidence below which boxes won't be drawn.")]
    public float MinBoxConfidence = 0.3f;


    NNHandler nn;
    YOLOHandler yolo;

    WebCamTextureProvider CamTextureProvider;

    Color[] colorArray = new Color[] { Color.red, Color.green, Color.blue, Color.cyan, Color.magenta, Color.yellow };

    void OnEnable()
    {
        nn = new NNHandler(ModelFile);
        yolo = new YOLOHandler(nn);

        var firstInput = nn.model.inputs[0];
        int height = firstInput.shape[5];
        int width = firstInput.shape[6];

        CamTextureProvider = new WebCamTextureProvider(width, height);
        CamTextureProvider.Start();

        YOLOv8SegmentationPostprocessor.DiscardThreshold = MinBoxConfidence;
    }

    void Update()
    {
        Texture2D texture = GetNextTexture();

        var boxes = yolo.Run(texture);
        DrawResults(boxes, texture);
        ImageUI.texture = texture;
    }

    Texture2D GetNextTexture()
    {
        return CamTextureProvider.GetTexture();
    }

    private void OnDisable()
    {
        nn.Dispose();
        CamTextureProvider.Stop();
    }

    private void DrawResults(IEnumerable<ResultBoxWithMasks> results, Texture2D img)
    {
        results.ForEach(box => DrawBox(box, img));
    }

    private void DrawBox(ResultBoxWithMasks box, Texture2D img)
    {
        Color boxColor = colorArray[box.bestClassIndex % colorArray.Length];
        int boxWidth = (int)(box.score / MinBoxConfidence);
        TextureTools.DrawRectOutline(img, box.rect, boxColor, boxWidth, rectIsNormalized: false, revertY: true);

        const float maskFactor = 0.25f;
        Tensor imgTensor = new(img);
        Tensor factorTensor = new(1, 3, new[] { boxColor.r * maskFactor, boxColor.g * maskFactor, boxColor.b * maskFactor });
        Tensor mask = box.masks;
        Tensor colorMask = yolo.ops.Mul(new[] { mask, factorTensor });
        mask.tensorOnDevice.Dispose();
        factorTensor.tensorOnDevice.Dispose();
        Tensor imgWithMasks = yolo.ops.Add(new[] { imgTensor, colorMask });
        imgTensor.tensorOnDevice.Dispose();
        colorMask.tensorOnDevice.Dispose();
        var rt = imgWithMasks.ToRenderTexture();
        RenderTexture.active = rt;
        img.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
        img.Apply();
        imgWithMasks.tensorOnDevice.Dispose();
        RenderTexture.active = null;
        rt.Release();
    }
}
