using System;
using System.Collections;
using UnityEngine;
using UnityEngine.Profiling;

namespace Assets.Scripts
{
    public class WebCamTextureProvider
    {
        private Texture2D ResultTexture;
        private WebCamTexture webCamTexture;

        public WebCamTextureProvider(int width, int height, TextureFormat format = TextureFormat.RGB24, string cameraName = null)
        {
            ResultTexture = new Texture2D(width, height, format, mipChain: false);
            cameraName = cameraName != null ? cameraName : SelectCameraDevice();
            webCamTexture = new WebCamTexture(cameraName);
        }

        public void Start()
        {
            webCamTexture.Play();
        }

        public void Stop()
        {
            webCamTexture.Stop();
        }

        public Texture2D GetTexture()
        {
            return TextureTools.ResizeAndCropToCenter(webCamTexture, ref ResultTexture, ResultTexture.width, ResultTexture.height);
        }

        /// <summary>
        /// Return first backfaced camera name if avaible, otherwise first possible
        /// </summary>
        private string SelectCameraDevice()
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
}