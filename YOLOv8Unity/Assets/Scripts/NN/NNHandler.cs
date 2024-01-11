using Unity.Barracuda;

public class NNHandler : System.IDisposable
{
    public Model model;
    public IWorker worker;

    public NNHandler(NNModel nnmodel)
    {
        model = ModelLoader.Load(nnmodel);
        worker = WorkerFactory.CreateWorker(model);
    }

    public void Dispose()
    {
        worker.Dispose();
    }
}
