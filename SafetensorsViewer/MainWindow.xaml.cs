using CommunityToolkit.Mvvm.Input;
using Microsoft.Win32;
using Onnxify.Safetensors;
using SciChart.Charting2D.Interop;
using System.ComponentModel;
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

namespace SafetensorsViewer
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    ///


    public partial class MainWindow : Window, INotifyPropertyChanged, INotifyPropertyChanging
    {
        SafeTensors? _safetensors;
        SafeTensors? LoadedSafetensors {
            get {
                return _safetensors;
            } set {
                if (_safetensors != value) {
                    PropertyChanging?.Invoke(this, new PropertyChangingEventArgs(nameof(LoadedSafetensors)));
                    _safetensors = value;
                    PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(LoadedSafetensors)));
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
                Task<SafeTensors> loadTask = SafeTensors.LoadFromFileAsync(openFileDialog.FileName);
                LoadedSafetensors = await loadTask;
            }
        }
    }
}