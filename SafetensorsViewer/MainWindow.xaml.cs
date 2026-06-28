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
using Onnxify.Safetensors;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Win32;

namespace SafetensorsViewer
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    ///


    public partial class MainWindow : Window
    {
        public RelayCommand OpenCommand;
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
                SafeTensors T = await loadTask;
                int a = 0;
                // Do something with the loaded tensors
            }
        }
    }
}