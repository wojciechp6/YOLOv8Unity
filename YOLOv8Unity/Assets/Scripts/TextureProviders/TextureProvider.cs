using System;
using System.Collections;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using UnityEngine;
using UnityEngine.Profiling;
using UnityEngine.Video;

namespace Assets.Scripts.TextureProviders
{
    [Serializable]
    public abstract class TextureProvider
    {
        protected Texture2D ResultTexture;
        protected Texture InputTexture;

        public TextureProvider(int width, int height, TextureFormat format = TextureFormat.RGB24)
        {
            ResultTexture = new Texture2D(width, height, format, mipChain: false);
        }

        ~TextureProvider()
        {
            Stop();
        }

        public abstract void Start();

        public abstract void Stop();

        public virtual Texture2D GetTexture()
        {
            return TextureTools.ResizeAndCropToCenter(InputTexture, ref ResultTexture, ResultTexture.width, ResultTexture.height);
        }

        public abstract TextureProviderType.ProviderType TypeEnum();
    }


    public static class TextureProviderType
    {
        static TextureProvider[] providers;

        static TextureProviderType()
        {
            providers = new TextureProvider[]{ 
                RuntimeHelpers.GetUninitializedObject(typeof(WebCamTextureProvider)) as WebCamTextureProvider,
                RuntimeHelpers.GetUninitializedObject(typeof(VideoTextureProvider)) as VideoTextureProvider };
        }

        public enum ProviderType
        {
            WebCam,
            Video
        }

        static public Type GetProviderType(ProviderType type)
        {
            foreach(var provider in providers)
            {
                if (provider.TypeEnum() == type)
                    return provider.GetType();
            }
            throw new InvalidEnumArgumentException();
        }
    }
}