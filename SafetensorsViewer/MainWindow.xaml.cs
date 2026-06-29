using CommunityToolkit.Mvvm.Input;
using Microsoft.Win32;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using TorchSharp;
using TorchSharp.Utils;
using static Tensorboard.CostGraphDef.Types;

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
                    MyPlot.Reset();
                    MyPlot.Plot.ShowLegend();
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
                if (_selectedTensorKey != null)
                {
                    PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(SelectedTensorKey)));

                    Dictionary<string, torch.Tensor> t = TorchSharp.PyBridge.Safetensors.LoadStateDict(tensorpath, [_selectedTensorKey]);
                    if (!t.ContainsKey(_selectedTensorKey))
                    {
                        _selectedTensorKey = _selectedTensorKey + ".weight";
                    }
                    if (t.ContainsKey(_selectedTensorKey))
                    {
                        using var doubleTensor = t.Values.Last().to_type(torch.ScalarType.Float64);
                        TensorAccessor<double> vv = doubleTensor.data<double>();

                        double[,] nativeMatrix = new double[t[_selectedTensorKey].shape.Count() > 0 ? t[_selectedTensorKey].shape[0] : 1, t[_selectedTensorKey].shape.Count() > 1 ? t[_selectedTensorKey].shape[1] : 1];
                        var targetSpan = MemoryMarshal.CreateSpan(ref nativeMatrix[0, 0], nativeMatrix.GetLength(0) * nativeMatrix.GetLength(1));
                        vv.CopyTo(targetSpan);

                        DataMatrix = nativeMatrix;
                    }
                }
                else {
                    DataMatrix = null;
                }
            }
        }
        public ObservableCollection<TensorNodeViewModel> TensorKeys { get; set; } = new ObservableCollection<TensorNodeViewModel>();
        public RelayCommand? OpenCommand;

        public event PropertyChangedEventHandler? PropertyChanged;
        public event PropertyChangingEventHandler? PropertyChanging;

        public ICommand openCMD => OpenCommand ??= new RelayCommand(CommandOpen, () => true);

        private bool _isHeatmapChecked = true;
        private bool _isHistogramChecked;
        private bool _isStatisticsChecked;

        public bool IsHeatmapChecked
        {
            get => _isHeatmapChecked;
            set
            {
                if (_isHeatmapChecked != value)
                {
                    _isHeatmapChecked = value;
                    PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(IsHeatmapChecked)));
                    if (value)
                    {
                        IsHistogramChecked = false;
                        IsStatisticsChecked = false;
                    }
                    else if (!IsHistogramChecked && !IsStatisticsChecked)
                    {
                        _isHeatmapChecked = true;
                        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(IsHeatmapChecked)));
                    }
                }
            }
        }

        public bool IsHistogramChecked
        {
            get => _isHistogramChecked;
            set
            {
                if (_isHistogramChecked != value)
                {
                    _isHistogramChecked = value;
                    PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(IsHistogramChecked)));
                    if (value)
                    {
                        IsHeatmapChecked = false;
                        IsStatisticsChecked = false;
                    }
                    else if (!IsHeatmapChecked && !IsStatisticsChecked)
                    {
                        _isHistogramChecked = true;
                        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(IsHistogramChecked)));
                    }
                }
            }
        }

        public bool IsStatisticsChecked
        {
            get => _isStatisticsChecked;
            set
            {
                if (_isStatisticsChecked != value)
                {
                    _isStatisticsChecked = value;
                    PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(IsStatisticsChecked)));
                    if (value)
                    {
                        IsHeatmapChecked = false;
                        IsHistogramChecked = false;
                    }
                    else if (!IsHeatmapChecked && !IsHistogramChecked)
                    {
                        _isStatisticsChecked = true;
                        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(IsStatisticsChecked)));
                    }
                }
            }
        }

        private RelayCommand<string>? _viewCommand;
        public ICommand viewCMD => _viewCommand ??= new RelayCommand<string>(CommandView);

        private void CommandView(string? viewType)
        {
            if (string.IsNullOrEmpty(viewType)) return;
            if (viewType == "Heatmap") IsHeatmapChecked = true;
            else if (viewType == "Histogram") IsHistogramChecked = true;
            else if (viewType == "Statistics") IsStatisticsChecked = true;
        }

        public MainWindow()
        {
            InitializeComponent();
            TensorNodeViewModel.NodeSelected += OnNodeSelected;
        }

        private void OnNodeSelected(TensorNodeViewModel selectedNode)
        {
            if (selectedNode.Tag != null)
            {
                SelectedTensorKey = selectedNode.Tag;
            }
        }

        async void CommandOpen()
        {


            OpenFileDialog openFileDialog = new();
            if (openFileDialog.ShowDialog() == true)
            {
                tensorpath = openFileDialog.FileName;
                SafetensorsFileReader sfr = new(tensorpath);

                TensorKeys.Clear();

                Dictionary<string, TensorNodeViewModel> nodes = new();
                TensorNodeViewModel? parent;
                foreach (string key in sfr.Keys)
                {
                    string[] parts = key.Split('.');
                    string path = "";

                    parent = null;

                    foreach (string part in parts)
                    {
                        path = path.Length == 0 ? part : $"{path}.{part}";

                        if (!nodes.TryGetValue(path, out TensorNodeViewModel? node))
                        {
                            node = new TensorNodeViewModel { Header = part };
                            nodes[path] = node;

                            if (parent == null)
                                TensorKeys.Add(node);
                            else
                                parent.Children.Add(node);
                        }

                        parent = node;
                    }
                    if (parent != null)
                    {
                        parent.Tag = key;
                    }
                }
            }
        }
    }
}