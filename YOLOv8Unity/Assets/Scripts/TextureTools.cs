using System;
using System.Threading.Tasks;
using System.Xml.Linq;
using Unity.Barracuda;
using UnityEngine;
using UnityEngine.UIElements;

class TextureTools
{
    public static Texture2D ResizeAndCropToCenter(Texture texture, ref Texture2D result, int width, int height)
    {
        float widthRatio = width / (float)texture.width;
        float heightRatio = height / (float)texture.height;
        float ratio = widthRatio > heightRatio ? widthRatio : heightRatio;

        Vector2Int renderTexturetSize = new((int)(texture.width * ratio), (int)(texture.height * ratio));
        RenderTexture renderTexture = RenderTexture.GetTemporary(renderTexturetSize.x, renderTexturetSize.y);
        Graphics.Blit(texture, renderTexture);
        
        RenderTexture previousRenderTexture = RenderTexture.active;
        RenderTexture.active = renderTexture;

        int xOffset = (renderTexturetSize.x - width) / 2;
        int yOffset = (renderTexturetSize.y - width) / 2;
        result.ReadPixels(new Rect(xOffset, yOffset, width, height), destX: 0, destY: 0);
        result.Apply();
        
        RenderTexture.active = previousRenderTexture;
        RenderTexture.ReleaseTemporary(renderTexture);
        return result;
    }

    /// <summary>
    /// Draw rectange outline on texture
    /// </summary>
    /// <param name="width">Width of outline</param>
    /// <param name="rectIsNormalized">Are rect values normalized?</param>
    /// <param name="revertY">Pass true if y axis has opposite direction than texture axis</param>
    public static void DrawRectOutline(Texture2D texture, Rect rect, Color color, int width = 1, bool rectIsNormalized = true, bool revertY = false)
    {
        if (rectIsNormalized)
        {
            rect.x *= texture.width;
            rect.y *= texture.height;
            rect.width *= texture.width;
            rect.height *= texture.height;
        }

        if (revertY)
            rect.y = rect.y * -1 + texture.height - rect.height;

        if (rect.width <= 0 || rect.height <= 0)
            return;

        DrawRect(texture, rect.x, rect.y, rect.width + width, width, color);
        DrawRect(texture, rect.x, rect.y + rect.height, rect.width + width, width, color);

        DrawRect(texture, rect.x, rect.y, width, rect.height + width, color);
        DrawRect(texture, rect.x + rect.width, rect.y, width, rect.height + width, color);
        texture.Apply(); 
    }

    static private void DrawRect(Texture2D texture, float x, float y, float width, float height, Color color)
    {
        if (x > texture.width || y > texture.height)
            return;

        if (x < 0)
        {
            width += x;
            x = 0;
        }
        if (y < 0)
        {
            height += y;
            y = 0;
        }

        width = x + width > texture.width ? texture.width - x : width;
        height = y + height > texture.height ? texture.height - y : height;

        x = (int)x;
        y = (int)y;
        width = (int)width;
        height = (int)height;

        if (width <= 0 || height <= 0)
            return;

        int pixelsCount = (int)width * (int)height;
        Color32[] colors = new Color32[pixelsCount];
        Array.Fill(colors, color);

        texture.SetPixels32((int)x, (int)y, (int)width, (int)height, colors);
    }

    public static void RenderMaskOnTexture(Tensor mask, Texture2D texture, Color color, float maskFactor = 0.25f)
    {
        IOps ops = BarracudaUtils.CreateOps(WorkerFactory.Type.ComputePrecompiled);
        Tensor imgTensor = new(texture);
        Tensor factorTensor = new(1, 3, new[] { color.r * maskFactor, color.g * maskFactor, color.b * maskFactor });
        Tensor colorMask = ops.Mul(new[] { mask, factorTensor });
        Tensor imgWithMasks = ops.Add(new[] { imgTensor, colorMask });

        RenderTensorToTexture(imgWithMasks, texture);

        factorTensor.tensorOnDevice.Dispose();
        imgTensor.tensorOnDevice.Dispose();
        colorMask.tensorOnDevice.Dispose();
        imgWithMasks.tensorOnDevice.Dispose();
    }

    private static void RenderTensorToTexture(Tensor tensor, Texture2D texture)
    {
        RenderTexture renderTexture = tensor.ToRenderTexture();
        RenderTexture.active = renderTexture;
        texture.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
        texture.Apply();
        RenderTexture.active = null;
        renderTexture.Release();
    }
}

