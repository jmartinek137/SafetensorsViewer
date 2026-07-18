using CommunityToolkit.Mvvm.Input;
using Microsoft.Win32;
using ScottPlot;
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

        Dictionary<string, SafetensorsDType> _originalDTypes = new();

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

        bool _isEditModeEnabled;
        public bool IsEditModeEnabled
        {
            get => _isEditModeEnabled;
            set
            {
                if (_isEditModeEnabled != value)
                {
                    _isEditModeEnabled = value;
                    PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(IsEditModeEnabled)));
                    UpdateEditModeState();
                }
            }
        }

        double _brushStep = 1e-5;
        public double BrushStep
        {
            get => _brushStep;
            set
            {
                if (_brushStep != value)
                {
                    _brushStep = value;
                    PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(BrushStep)));
                }
            }
        }

        public Dictionary<string, List<TensorEdit>> PendingEditsByTensor { get; } = new Dictionary<string, List<TensorEdit>>();

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

                    SafetensorsFileReader sfr = new(tensorpath);
                    SafetensorsDType originalDType = SafetensorsDTypeExtensions.Parse(sfr.GetInfo(_selectedTensorKey).DType);
                    _originalDTypes[_selectedTensorKey] = originalDType;

                    torch.Tensor tensor = sfr.LoadTensor(_selectedTensorKey);

                    // Replay any pending edits for this tensor.
                    if (PendingEditsByTensor.TryGetValue(_selectedTensorKey, out List<TensorEdit>? edits))
                    {
                        foreach (TensorEdit edit in edits)
                        {
                            tensor[edit.Y, edit.X] = torch.tensor(edit.NewValue);
                        }
                    }

                    LoadedSafetensors = tensor;
                    TensorAccessor<double> vv = tensor.data<double>();
                    double[,] nativeMatrix = new double[tensor.shape.Count() > 0 ? tensor.shape[0] : 1, tensor.shape.Count() > 1 ? tensor.shape[1] : 1];
                    Span<double> targetSpan = MemoryMarshal.CreateSpan(ref nativeMatrix[0, 0], nativeMatrix.GetLength(0) * nativeMatrix.GetLength(1));
                    vv.CopyTo(targetSpan);
                    DataMatrix = nativeMatrix;

                    HasChanges = edits is { Count: > 0 };
                    Status = "Ready";
                }
                else
                {
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

            MyPlot.MouseMove += MyPlot_MouseMove;
            MyPlot.MouseDown += MyPlot_MouseDown;
            UpdateEditModeState();
        }

        void UpdateEditModeState()
        {
            if (IsEditModeEnabled)
            {
                MyPlot.UserInputProcessor.Disable();
            }
            else
            {
                MyPlot.UserInputProcessor.Enable();
            }
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
            if (string.IsNullOrEmpty(tensorpath) || LoadedSafetensors is null)
                return;

            Status = "Saving file";
            try
            {
                SaveTensorTo(tensorpath);
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
            if (string.IsNullOrEmpty(tensorpath) || LoadedSafetensors is null)
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
                    SaveTensorTo(saveFileDialog.FileName);
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

        void SaveTensorTo(string filePath)
        {
            if (LoadedSafetensors is null || SelectedTensorKey is null)
                return;

            SafetensorsDType originalDType = _originalDTypes.GetValueOrDefault(SelectedTensorKey, SafetensorsDType.F64);
            SafetensorsFileWriter.SaveTensor(filePath, SelectedTensorKey, LoadedSafetensors.clone(), originalDType);
            PendingEditsByTensor.Remove(SelectedTensorKey);
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
                PendingEditsByTensor.Clear();
                _originalDTypes.Clear();
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

        bool _isPainting;

        void MyPlot_MouseDown(object sender, MouseButtonEventArgs e)
        {
            if (!IsEditModeEnabled || DataMatrix == null)
                return;

            if (e.ChangedButton == MouseButton.Left)
            {
                _isPainting = true;
                ApplyBrushAtMousePosition(e.GetPosition(MyPlot), increase: true);
                e.Handled = true;
            }
            else if (e.ChangedButton == MouseButton.Right)
            {
                _isPainting = true;
                ApplyBrushAtMousePosition(e.GetPosition(MyPlot), increase: false);
                e.Handled = true;
            }
        }

        void MyPlot_MouseMove(object sender, MouseEventArgs e)
        {
            if (!_isPainting || !IsEditModeEnabled || DataMatrix == null)
                return;

            bool increase = e.LeftButton == MouseButtonState.Pressed;
            bool decrease = e.RightButton == MouseButtonState.Pressed;

            if (increase || decrease)
            {
                ApplyBrushAtMousePosition(e.GetPosition(MyPlot), increase);
                e.Handled = true;
            }
            else
            {
                _isPainting = false;
            }
        }

        void MyPlot_MouseUp(object sender, MouseButtonEventArgs e)
        {
            _isPainting = false;
        }

        void ApplyBrushAtMousePosition(Point mousePosition, bool increase)
        {
            Pixel mousePixel = new(mousePosition.X, mousePosition.Y);
            Coordinates coords = MyPlot.Plot.GetCoordinates(mousePixel);

            int x = (int)Math.Round(coords.X);
            int y = (int)Math.Round(coords.Y);

            int rows = DataMatrix!.GetLength(0);
            int cols = DataMatrix.GetLength(1);

            if (x < 0 || x >= cols || y < 0 || y >= rows)
                return;

            int dataY = rows - 1 - y;

            double delta = increase ? BrushStep : -BrushStep;
            double newValue = DataMatrix[dataY, x] + delta;
            DataMatrix[dataY, x] = newValue;

            // Keep the underlying tensor in sync so the edit is visible immediately.
            if (LoadedSafetensors is not null)
            {
                LoadedSafetensors[dataY, x] = torch.tensor(newValue);
            }

            if (SelectedTensorKey is not null)
            {
                if (!PendingEditsByTensor.TryGetValue(SelectedTensorKey, out List<TensorEdit>? edits))
                {
                    edits = new List<TensorEdit>();
                    PendingEditsByTensor[SelectedTensorKey] = edits;
                }
                edits.Add(new TensorEdit(x, dataY, newValue));
            }

            HasChanges = true;

            heatmap!.Intensities = DataMatrix;
            heatmap.Update();
            MyPlot.Refresh();

            // Refresh histogram to reflect the new value distribution.
            MyPlot1.Plot.Clear();
            var histogram = ScottPlot.Statistics.Histogram.WithBinCount(
                (int)Math.Ceiling(Math.Sqrt(_dataMatrix.Length)),
                torch.min(LoadedSafetensors).item<double>() - 1e-5,
                torch.max(LoadedSafetensors).item<double>() + 1e-5);
            histogram.AddRange(_dataMatrix.Cast<double>());
            MyPlot1.Plot.Add.Histogram(histogram);
            MyPlot1.Plot.XLabel("Value");
            MyPlot1.Plot.YLabel("Count");
            MyPlot1.Refresh();
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

    public record TensorEdit(int X, int Y, double NewValue);
}
