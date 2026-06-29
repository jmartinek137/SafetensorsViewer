using System;
using System.Collections.ObjectModel;
using System.ComponentModel;

namespace SafetensorsViewer
{
    public class TensorNodeViewModel : INotifyPropertyChanged
    {
        private string _header = string.Empty;
        private string? _tag;
        private bool _isSelected;

        public string Header
        {
            get => _header;
            set
            {
                if (_header != value)
                {
                    _header = value;
                    OnPropertyChanged(nameof(Header));
                }
            }
        }

        public string? Tag
        {
            get => _tag;
            set
            {
                if (_tag != value)
                {
                    _tag = value;
                    OnPropertyChanged(nameof(Tag));
                }
            }
        }

        public bool IsSelected
        {
            get => _isSelected;
            set
            {
                if (_isSelected != value)
                {
                    _isSelected = value;
                    OnPropertyChanged(nameof(IsSelected));
                    if (_isSelected)
                    {
                        NodeSelected?.Invoke(this);
                    }
                }
            }
        }

        public ObservableCollection<TensorNodeViewModel> Children { get; } = new();

        public static event Action<TensorNodeViewModel>? NodeSelected;

        public event PropertyChangedEventHandler? PropertyChanged;
        protected virtual void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}
