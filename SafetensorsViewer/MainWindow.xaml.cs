using CommunityToolkit.Mvvm.Input;
using Microsoft.Win32;
using ScottPlot.Plottables;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.IO;
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
        Heatmap? heatmap;

        string _status = string.Empty;
        public string Status
        {
            get => _status;
            set
            {
                if (_status != value)
                {
                    _status = value;
                    PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(Status)));
                }
            }
        }

        bool _hasChanges;
        public bool HasChanges
        {
            get => _hasChanges;
            set
            {
                if (_hasChanges != value)
                {
                    _hasChanges = value;
                    PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(HasChanges)));
                }
            }
        }

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
                    if (value != null)
                    {

                            heatmap ??= MyPlot.Plot.Add.Heatmap(value);
                            heatmap.Intensities = value;
                            heatmap.Update();
                            MyPlot.Refresh();

                        MyPlot1.Plot.Clear();
                        var histogram = ScottPlot.Statistics.Histogram.WithBinCount((int)Math.Ceiling(Math.Sqrt(_dataMatrix.Length)), torch.min(LoadedSafetensors).item<double>() - 1e-5, torch.max(LoadedSafetensors).item<double>() + 1e-5);
                        histogram.AddRange(_dataMatrix.Cast<double>());
                        var histogramPlot = MyPlot1.Plot.Add.Histogram(histogram);
                        MyPlot1.Plot.XLabel("Value");
                        MyPlot1.Plot.YLabel("Count");
                        MyPlot1.Refresh();


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
                    if (t.TryGetValue(_selectedTensorKey, out torch.Tensor? value1))
                    {
                        torch.Tensor doubleTensor = value1.to_type(torch.ScalarType.Float64);
                        TensorAccessor<double> vv = doubleTensor.data<double>();
                        LoadedSafetensors = doubleTensor;
                        double[,] nativeMatrix = new double[t[_selectedTensorKey].shape.Count() > 0 ? t[_selectedTensorKey].shape[0] : 1, t[_selectedTensorKey].shape.Count() > 1 ? t[_selectedTensorKey].shape[1] : 1];
                        Span<double> targetSpan = MemoryMarshal.CreateSpan(ref nativeMatrix[0, 0], nativeMatrix.GetLength(0) * nativeMatrix.GetLength(1));
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
        public RelayCommand? SaveCommand;
        public RelayCommand? SaveAsCommand;

        public event PropertyChangedEventHandler? PropertyChanged;
        public event PropertyChangingEventHandler? PropertyChanging;

        public ICommand openCMD => OpenCommand ??= new RelayCommand(CommandOpen, () => true);
        public ICommand saveCMD => SaveCommand ??= new RelayCommand(CommandSave, () => CanSave);
        public ICommand saveAsCMD => SaveAsCommand ??= new RelayCommand(CommandSaveAs, () => CanSaveAs);

        bool CanSave => !string.IsNullOrEmpty(tensorpath) && HasChanges;
        bool CanSaveAs => !string.IsNullOrEmpty(tensorpath);

        public MainWindow()
        {
            InitializeComponent();
            heatmap ??= MyPlot.Plot.Add.Heatmap(new double[,] { { 0.0 } });
            heatmap.Colormap = new ScottPlot.Colormaps.Turbo();
            MyPlot.Plot.Add.ColorBar(heatmap);
            MyPlot.Plot.ShowLegend();
            Status = "Ready";

            PropertyChanged += (s, e) =>
            {
                if (e.PropertyName == nameof(HasChanges))
                {
                    SaveCommand?.NotifyCanExecuteChanged();
                    SaveAsCommand?.NotifyCanExecuteChanged();
                }
            };
        }
        bool PromptSaveUnsavedChanges()
        {
            if (!HasChanges)
                return true;

            MessageBoxResult result = MessageBox.Show(
                "There are unsaved changes. Do you want to save them?",
                "Unsaved changes",
                MessageBoxButton.YesNoCancel,
                MessageBoxImage.Warning);

            if (result == MessageBoxResult.Cancel)
                return false;

            if (result == MessageBoxResult.Yes)
            {
                CommandSave();
                return !HasChanges;
            }

            return true;
        }

        void CommandSave()
        {
            if (string.IsNullOrEmpty(tensorpath))
                return;

            Status = "Saving file";
            try
            {
                // For now, simply overwrite the existing file. Once editing is supported,
                // this should serialize the modified tensors back to safetensors format.
                File.Copy(tensorpath, tensorpath, overwrite: true);
                HasChanges = false;
                Status = "Ready";
            }
            catch (Exception ex)
            {
                Status = "Save failed";
                MessageBox.Show($"Could not save file: {ex.Message}", "Save error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        void CommandSaveAs()
        {
            if (string.IsNullOrEmpty(tensorpath))
                return;

            SaveFileDialog saveFileDialog = new()
            {
                Filter = "Safetensors files (*.safetensors)|*.safetensors|All files (*.*)|*.*",
                FileName = Path.GetFileName(tensorpath)
            };

            if (saveFileDialog.ShowDialog() == true)
            {
                Status = "Saving file as";
                try
                {
                    File.Copy(tensorpath, saveFileDialog.FileName, overwrite: true);
                    tensorpath = saveFileDialog.FileName;
                    SaveCommand?.NotifyCanExecuteChanged();
                    SaveAsCommand?.NotifyCanExecuteChanged();
                    HasChanges = false;
                    Title = $"SafetensorsViewer - {Path.GetFileName(tensorpath)}";
                    Status = "Ready";
                }
                catch (Exception ex)
                {
                    Status = "Save failed";
                    MessageBox.Show($"Could not save file: {ex.Message}", "Save error", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }

        async void CommandOpen()
        {
            if (!PromptSaveUnsavedChanges())
                return;

            OpenFileDialog openFileDialog = new()
            {
                Filter = "Safetensors files (*.safetensors)|*.safetensors|All files (*.*)|*.*"
            };
            if (openFileDialog.ShowDialog() == true)
            {
                Status = "Loading safetensors file";
                tensorpath = openFileDialog.FileName;
                SaveCommand?.NotifyCanExecuteChanged();
                SaveAsCommand?.NotifyCanExecuteChanged();
                Title = $"SafetensorsViewer - {Path.GetFileName(tensorpath)}";
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
                HasChanges = false;
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

        void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            if (!PromptSaveUnsavedChanges())
                e.Cancel = true;
        }

        void ExitMenuItem_Click(object sender, RoutedEventArgs e)
        {
            Close();
        }
    }
}