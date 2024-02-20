using NN;
using System.Collections;
using UnityEngine;

namespace Assets.Scripts
{
    public class Segmentator : Detector
    {
        YOLOv8Segmentation yolo;

        // Use this for initialization
        void OnEnable()
        {
            nn = new NNHandler(ModelFile);
            yolo = new YOLOv8Segmentation(nn);

            textureProvider = GetTextureProvider(nn.model);
            textureProvider.Start();
        }

        // Update is called once per frame
        void Update()
        {
            YOLOv8OutputReader.DiscardThreshold = MinBoxConfidence;
            Texture2D texture = GetNextTexture();

            var boxes = yolo.Run(texture);
            DrawResults(boxes, texture);
            ImageUI.texture = texture;
        }

        void OnDisable()
        {
            nn.Dispose(); 
            textureProvider.Stop();
        }

        protected override void DrawBox(ResultBox box, Texture2D img)
        {
            base.DrawBox(box, img);

            ResultBoxWithMask boxWithMask = box as ResultBoxWithMask;
            Color boxColor = colorArray[box.bestClassIndex % colorArray.Length];
            TextureTools.RenderMaskOnTexture(boxWithMask.masks, img, boxColor);
            boxWithMask.masks.tensorOnDevice.Dispose();
        }
    }
}