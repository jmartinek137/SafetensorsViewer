using CommunityToolkit.Mvvm.Input;
using Microsoft.Win32;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using TorchSharp;
using TorchSharp.PyBridge;
using TorchSharp.Utils;

namespace SafetensorsViewer
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    ///


    public partial class MainWindow : Window, INotifyPropertyChanged, INotifyPropertyChanging
    {
        torch.Tensor? _safetensors;
        torch.Tensor? LoadedSafetensors {
            get {
                return _safetensors;
            } set {
                if (!EqualityComparer<torch.Tensor?>.Default.Equals(_safetensors, value))
                {
                    PropertyChanging?.Invoke(this, new PropertyChangingEventArgs(nameof(LoadedSafetensors)));
                    _safetensors = value;
                    PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(LoadedSafetensors)));

                }
            }
        }
        double[,]? _dataMatrix;
        public double[,]? DataMatrix {
            get {
                return _dataMatrix;
            } set {
                if (_dataMatrix != value) {
                    PropertyChanging?.Invoke(this, new PropertyChangingEventArgs(nameof(DataMatrix)));
                    _dataMatrix = value;
                    PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(DataMatrix)));
                    MyPlot.Plot.Clear();
                    if (value != null) {
                        var heatmap = MyPlot.Plot.Add.Heatmap(value);
                        heatmap.Colormap = new ScottPlot.Colormaps.Turbo();
                        MyPlot.Plot.Add.ColorBar(heatmap);
                        MyPlot.Refresh();
                    }
                }
            }
        }
        public RelayCommand? OpenCommand;

        public event PropertyChangedEventHandler? PropertyChanged;
        public event PropertyChangingEventHandler? PropertyChanging;

        public ICommand openCMD => OpenCommand ??= new RelayCommand(CommandOpen, () => true);

        public MainWindow()
        {
            InitializeComponent();
        }
        async void CommandOpen()
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "SafeTensors files (*.safetensors, *.sft)|*.safetensors;*.sft|All files (*.*)|*.*";
            if (openFileDialog.ShowDialog() == true)
            {
                SafetensorsFileReader sfr = new SafetensorsFileReader(openFileDialog.FileName);
                long[] shape = sfr.TensorRegistry[sfr.Keys.Last()].Shape;
                int height = (int)shape[0];
                int width = (int)shape[1];
                byte[] v = sfr.GetTensor(sfr.Keys.Last());
                torch.Tensor tensor = torch.tensor(v, shape, sfr.TensorRegistry[sfr.Keys.Last()].DType);
                TensorAccessor<double> vv = tensor.to_type(torch.ScalarType.Float64).data<double>();
                double[,] nativeMatrix = new double[height, width];
                var targetSpan = MemoryMarshal.CreateSpan(ref nativeMatrix[0, 0], height * width);
                vv.CopyTo(targetSpan);
                DataMatrix = nativeMatrix;
            }
        }
    }
}