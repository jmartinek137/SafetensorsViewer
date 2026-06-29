using CommunityToolkit.Mvvm.Input;
using Microsoft.Win32;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Input;
using TorchSharp;
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
        string tensorpath=string.Empty;
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
        private string? _selectedTensorKey;

        public string? SelectedTensorKey
        {
            get => _selectedTensorKey;
            set
            {
                if (_selectedTensorKey == value)
                    return;

                _selectedTensorKey = value;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(SelectedTensorKey)));

                Dictionary<string, torch.Tensor> t = TorchSharp.PyBridge.Safetensors.LoadStateDict(tensorpath, [_selectedTensorKey]);
                using var doubleTensor = t.Values.Last().to_type(torch.ScalarType.Float64);
                TensorAccessor<double> vv = doubleTensor.data<double>();

                double[,] nativeMatrix = new double[t[_selectedTensorKey].shape[0], t[_selectedTensorKey].shape.Count() > 1 ? t[_selectedTensorKey].shape[1] : 1];
                var targetSpan = MemoryMarshal.CreateSpan(ref nativeMatrix[0, 0], nativeMatrix.GetLength(0) * nativeMatrix.GetLength(1));
                vv.CopyTo(targetSpan);

                DataMatrix = nativeMatrix;
            }
        }
        public ObservableCollection<string> TensorKeys { get; set; } = new ObservableCollection<string>();
        public RelayCommand? OpenCommand;

        public event PropertyChangedEventHandler? PropertyChanged;
        public event PropertyChangingEventHandler? PropertyChanging;

        public ICommand openCMD => OpenCommand ??= new RelayCommand(CommandOpen, () => true);

        public MainWindow()
        {
            InitializeComponent();
            int[] t = new int[] { 0x7Fffffff, 1, 0, 0};
            var tv = torch.frombuffer(t, torch.ScalarType.Int32,4,0);

        }
        async void CommandOpen()
        {


            OpenFileDialog openFileDialog = new OpenFileDialog();
            if (openFileDialog.ShowDialog() == true)
            {
                tensorpath = openFileDialog.FileName;
                SafetensorsFileReader sfr = new SafetensorsFileReader(tensorpath);
                foreach (var key in sfr.Keys)
                {
                    TensorKeys.Add(key);
                }
            }
        }
    }
}