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
    public NNModel modelFile;
    [Tooltip("Text file with classes names separated by coma ','")]
    public TextAsset classesFile;

    [Tooltip("RawImage component which will be used to draw resuls.")]
    public RawImage imageRenderer;

    [Range(0.0f, 1f)]
    [Tooltip("The minimum value of box confidence below which boxes won't be drawn.")]
    public float MinBoxConfidence = 0.3f;


    NNHandler nn;
    YOLOHandler yolo;

    WebCamTexture camTexture;
    Texture2D displayingTex;

    TextureScaler textureScaler;

    Color[] colorArray = new Color[] { Color.red, Color.green, Color.blue, Color.cyan, Color.magenta, Color.yellow };

    void OnEnable()
    {
        var dev = SelectCameraDevice();
        camTexture = new WebCamTexture(dev);
        camTexture.Play();

        nn = new NNHandler(modelFile);
        yolo = new YOLOHandler(nn);

        var firstInput = nn.model.inputs[0];
        int height = firstInput.shape[5];
        int width = firstInput.shape[6];
        textureScaler = new TextureScaler(width, height);
        
        YOLOv8Postprocessor.DiscardThreshold = MinBoxConfidence;
    }

    void Update()
    {
        CaptureAndPrepareTexture(camTexture, ref displayingTex);

        var boxes = yolo.Run(displayingTex);
        DrawResults(boxes, ref displayingTex);
        imageRenderer.texture = displayingTex;
    }

    private void OnDisable()
    {
        nn.Dispose();
        yolo.Dispose();
        textureScaler.Dispose();
        camTexture.Stop();
    }

    private void CaptureAndPrepareTexture(WebCamTexture camTexture, ref Texture2D tex)
    {
        Profiler.BeginSample("Texture processing");
        TextureCropTools.CropToSquare(camTexture, ref tex);
        textureScaler.Scale(tex);
        Profiler.EndSample();
    }

    private void DrawResults(IEnumerable<ResultBox> results, ref Texture2D img)
    {
        results.ForEach(box => DrawBox(box, ref displayingTex));
    }

    private void DrawBox(ResultBox box, ref Texture2D img)
    {
        Color boxColor = colorArray[box.bestClassIdx % colorArray.Length];
        int boxWidth = (int)(box.confidence / MinBoxConfidence);
        TextureDrawingUtils.DrawRect(img, box.rect, boxColor, boxWidth, rectIsNormalized: false, revertY: true);

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

    /// <summary>
    /// Return first backfaced camera name if avaible, otherwise first possible
    /// </summary>
    string SelectCameraDevice()
    {
        if (WebCamTexture.devices.Length == 0)
            throw new Exception("Any camera isn't avaible!");

        foreach (var cam in WebCamTexture.devices)
        {
            if (!cam.isFrontFacing)
                return cam.name;
        }
        return WebCamTexture.devices[0].name;
    }

}
