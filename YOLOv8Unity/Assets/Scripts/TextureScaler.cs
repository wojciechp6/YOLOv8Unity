using UnityEngine;
using UnityEngine.Profiling;

class TextureScaler : System.IDisposable
{
    // Based on https://gist.github.com/natsupy/e129936543f9b4663a37ea0762172b3b

    int width, height;
    RenderTexture renderTexture;

    /// <summary>
    /// TextureScaler scale texture to specified size
    /// </summary>
    /// <param name="width">Target new width of texture</param>
    /// <param name="height">Target new height of texture</param>
    public TextureScaler(int width, int height)
    {
        this.width = width;
        this.height = height;
        renderTexture = new RenderTexture(width, height, 32);
    }

    /// <summary>
    ///     Returns a scaled copy of given texture.
    /// </summary>
    /// <param name="src">Source texure to scale</param>
    /// <param name="mode">Filtering mode</param>
    public Texture2D Scaled(Texture2D src, FilterMode mode = FilterMode.Trilinear)
    {
        Profiler.BeginSample("TextureScaler.Scaled");
        Rect texR = new Rect(0, 0, width, height);
        _gpu_scale(src, mode);

        //Get rendered data back to a new texture
        Texture2D result = new Texture2D(width, height, TextureFormat.ARGB32, true);
        result.Reinitialize(width, height);
        result.ReadPixels(texR, 0, 0, true);

        Profiler.EndSample();
        return result;
    }

    /// <summary>
    /// Scales the texture data of the given texture.
    /// </summary>
    /// <param name="tex">Texure to scale</param>
    /// <param name="mode">Filtering mode</param>
    public void Scale(Texture2D tex, FilterMode mode = FilterMode.Trilinear)
    {
        Profiler.BeginSample("TextureScaler.Scale");

        Rect texR = new Rect(0, 0, width, height);
        _gpu_scale(tex, mode);

        // Update new texture
        tex.Reinitialize(width, height);
        tex.ReadPixels(texR, 0, 0, true);
        tex.Apply(true);        //Remove this if you hate us applying textures for you :)
        Profiler.EndSample();
    }

    // Internal unility that renders the source texture into the RTT - the scaling method itself.
    void _gpu_scale(Texture2D src, FilterMode fmode)
    {
        Profiler.BeginSample("TextureScaler.GpuScale");
        //We need the source texture in VRAM because we render with it
        src.filterMode = fmode;
        src.Apply(true);

        //Set the RTT in order to render to it
        Graphics.SetRenderTarget(renderTexture);

        //Setup 2D matrix in range 0..1, so nobody needs to care about sized
        GL.LoadPixelMatrix(0, 1, 1, 0);

        //Then clear & draw the texture to fill the entire RTT.
        GL.Clear(true, true, new Color(0, 0, 0, 0));
        Graphics.DrawTexture(new Rect(0, 0, 1, 1), src);
        Profiler.EndSample();
    }

    public void Dispose()
    {
        renderTexture.Release();
    }
}
