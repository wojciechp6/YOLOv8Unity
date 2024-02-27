using System;
using System.Collections;
using UnityEngine;
using UnityEngine.Video;

namespace Assets.Scripts.TextureProviders
{
    [Serializable]
    public class VideoTextureProvider : TextureProvider
    {
        [SerializeField]
        private VideoClip videoClip;
        [SerializeField]
        private bool loopClip = true;
        private VideoPlayer player;

        public VideoTextureProvider(int width, int height, TextureFormat format = TextureFormat.RGB24) : base(width, height, format)
        {
        }

        public VideoTextureProvider(VideoTextureProvider provider, int width, int height, TextureFormat format = TextureFormat.RGB24) : this(width, height, format)
        {
            if (provider == null)
                return;

            videoClip = provider.videoClip;
            loopClip = provider.loopClip;
        }

        public override void Start()
        {
            player = new GameObject("Video Player").AddComponent<VideoPlayer>();
            player.renderMode = VideoRenderMode.APIOnly;
            player.audioOutputMode = VideoAudioOutputMode.None;
            player.clip = videoClip;
        }

        public override void Stop()
        {
            GameObject.Destroy(player.gameObject);
        }

        public override TextureProviderType.ProviderType TypeEnum()
        {
            return TextureProviderType.ProviderType.Video;
        }

        public override Texture2D GetTexture()
        {
            bool reachedEnd = (ulong)player.frame == player.frameCount - 1;
            if (reachedEnd && loopClip)
                player.frame = 0;

            player.StepForward();
            InputTexture = player.texture ? player.texture : ResultTexture;
            return base.GetTexture();
        }
    }
}