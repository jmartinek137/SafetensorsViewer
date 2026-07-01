using CommunityToolkit.Mvvm.Input;
using Microsoft.Win32;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Controls;
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
                if (_safetensors is null || value is null || value.shape.SequenceCompareTo(_safetensors.shape) != 0 || !EqualityComparer<torch.Tensor?>.Default.Equals(_safetensors, value))
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
                if (_dataMatrix != value)
                {
                    Status = "Rendering data";
                    PropertyChanging?.Invoke(this, new PropertyChangingEventArgs(nameof(DataMatrix)));
                    _dataMatrix = value;
                    PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(DataMatrix)));
                    MyPlot.Reset();
                    MyPlot.Plot.ShowLegend();
                    if (value != null)
                    {
                        if (IsHeatmapChecked)
                        {
                            var heatmap = MyPlot.Plot.Add.Heatmap(value);
                            heatmap.Colormap = new ScottPlot.Colormaps.Turbo();
                            MyPlot.Plot.Add.ColorBar(heatmap);
                            MyPlot.Refresh();
                        }
                        else if (IsHistogramChecked)
                        {
                            var hist = ScottPlot.Statistics.Histogram.WithBinCount((int)Math.Ceiling(Math.Sqrt(_dataMatrix.Length)), torch.min(LoadedSafetensors).item<double>() - 1e-10, torch.max(LoadedSafetensors).item<double>() + 1e-10);
                            hist.AddRange(_dataMatrix.Cast<double>());
                            var histogram = MyPlot.Plot.Add.Histogram(hist);
                            MyPlot.Plot.XLabel("Value");
                            MyPlot.Plot.YLabel("Count");
                            MyPlot.Refresh();
                        }
                        else if (IsStatisticsChecked)
                        {
                            var mean = torch.mean(LoadedSafetensors).item<double>();
                            var median = torch.median(LoadedSafetensors).item<double>();
                            var stddev = torch.std(LoadedSafetensors).item<double>();
                            var var = torch.var(LoadedSafetensors).item<double>();
                            MyPlot.Plot.Add.Text($"Mean: {mean:F4}\nMedian: {median:F4}\nStd Dev: {stddev:F4}\nVariance: {var:F4}", 0.05, 0.95);
                        }
                    }
                    Status = "Ready";
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
                    Status = "Loading tensor";
                    PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(SelectedTensorKey)));

                    Dictionary<string, torch.Tensor> t = TorchSharp.PyBridge.Safetensors.LoadStateDict(tensorpath, [_selectedTensorKey]);
                    if (t.ContainsKey(_selectedTensorKey))
                    {
                        var doubleTensor = t[_selectedTensorKey].to_type(torch.ScalarType.Float64);
                        TensorAccessor<double> vv = doubleTensor.data<double>();
                        LoadedSafetensors = doubleTensor;
                        double[,] nativeMatrix = new double[t[_selectedTensorKey].shape.Count() > 0 ? t[_selectedTensorKey].shape[0] : 1, t[_selectedTensorKey].shape.Count() > 1 ? t[_selectedTensorKey].shape[1] : 1];
                        var targetSpan = MemoryMarshal.CreateSpan(ref nativeMatrix[0, 0], nativeMatrix.GetLength(0) * nativeMatrix.GetLength(1));
                        vv.CopyTo(targetSpan);
                        DataMatrix = nativeMatrix;
                    }
                    Status = "Ready";
                }
                else {
                    DataMatrix = null;
                }
            }
        }
        public ObservableCollection<TreeViewItem> TensorKeys { get; set; } = new ObservableCollection<TreeViewItem>();
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
        }
        async void CommandOpen()
        {


            OpenFileDialog openFileDialog = new();
            if (openFileDialog.ShowDialog() == true)
            {
                Status = "Loading safetensors file";
                tensorpath = openFileDialog.FileName;
                SafetensorsFileReader sfr = new(tensorpath);

                TensorKeys.Clear();
                Status = "Building tensor key tree";
                Dictionary<string, TreeViewItem> nodes = new();
                TreeViewItem? parent;
                foreach (string key in sfr.Keys)
                {
                    string[] parts = key.Split('.');
                    string path = "";

                    parent = null;

                    foreach (string part in parts)
                    {
                        path = path.Length == 0 ? part : $"{path}.{part}";

                        if (!nodes.TryGetValue(path, out TreeViewItem? node))
                        {
                            node = new TreeViewItem { Header = part };
                            nodes[path] = node;

                            if (parent == null)
                                TensorKeys.Add(node);
                            else
                                parent.Items.Add(node);
                        }

                        parent = node;
                    }
                    parent.Tag = key;
                }
                Status = "Ready";
            }
        }
        void TreeView_OnSelectedItemChanged(object sender, RoutedPropertyChangedEventArgs<object> e)
        {
            if (e.NewValue is TreeViewItem selectedKey)
            {
                SelectedTensorKey = selectedKey.Tag?.ToString();
            }
        }

        protected bool SetProperty<T>(ref T field, T newValue, [CallerMemberName] string propertyName = null)
        {
            if (!Equals(field, newValue))
            {
                field = newValue;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
                return true;
            }

            return false;
        }

        private string status;

        public string Status { get => status; set => SetProperty(ref status, value); }
    }
}