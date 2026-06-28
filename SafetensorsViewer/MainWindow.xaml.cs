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
                torch.Tensor loadTask = await Task.Run(() => torch.load_safetensors(openFileDialog.FileName));

                LoadedSafetensors = loadTask;
                MessageBox.Show($"Loaded {LoadedSafetensors.names.Count()} tensors.\nFirst tensor shape: {string.Join(", ", LoadedSafetensors.names)}\nData type: {LoadedSafetensors.dtype}\nData length: {LoadedSafetensors.numel() * LoadedSafetensors.element_size()} bytes", "SafeTensors Loaded", MessageBoxButton.OK, MessageBoxImage.Information);
            }
        }
    }
}