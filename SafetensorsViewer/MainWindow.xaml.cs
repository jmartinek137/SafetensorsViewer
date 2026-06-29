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
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(SelectedTensorKey)));

                Dictionary<string, torch.Tensor> t = TorchSharp.PyBridge.Safetensors.LoadStateDict(tensorpath, [_selectedTensorKey]);
                if(!t.ContainsKey(_selectedTensorKey))
                {
                    _selectedTensorKey = _selectedTensorKey+ ".weight";
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
        }
        public ObservableCollection<TreeViewItem> TensorKeys { get; set; } = new ObservableCollection<TreeViewItem>();
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


            OpenFileDialog openFileDialog = new();
            if (openFileDialog.ShowDialog() == true)
            {
                tensorpath = openFileDialog.FileName;
                SafetensorsFileReader sfr = new(tensorpath);

                TensorKeys.Clear();

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
            }
        }
        void TreeView_OnSelectedItemChanged(object sender, RoutedPropertyChangedEventArgs<object> e)
        {
            if (e.NewValue is TreeViewItem selectedKey)
            {
                SelectedTensorKey = selectedKey.Tag?.ToString();
            }
        }
    }
}