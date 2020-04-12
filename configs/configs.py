
def parse(p):
    """
    conf_file: name of the conf file to parse
    """
    p.add_argument('--train', action="store_true", default= False,
                        help='Train model or just show prediction')
    p.add_argument('--BLSTM', type=bool, default= False,
                        help='BLSTM model flag')

    p.add_argument('--ResNet', type=bool, default= False,
                        help='BLSTM model flag')
    p.add_argument('--attention', type=bool, default= False,
                        help='Resnet model flag')


    p.add_argument('--self_att', type=str, default= False,
                        help='Self attention model flag (only_self/heads)')

    p.add('--input_size', type=int, default=40, help='Input size')
    p.add('--hidden_size', type=int, default=300, help='Hidden size controlling FCN')
    p.add('--hidden_size_2', type=int, default=100, help='Hidden size controlling BLSTM')
    p.add('--num_classes', type=int, default=12, help='Number of output feature channels')
    p.add('--num_epochs', type=int, default=50, help='Number of epochs')
    p.add('--batch_size', type=int, default=2, help='Batch size')
    p.add('--learning_rate', type=float, default=1e-4, help='Learning rate')
    p.add('--weight_decay', type=float, default=1e-4, help='Weight decay')
    p.add('--emb_dim', type=int, default=128, help='?')
    p.add('--enc_hid_dim', type=int, default=128, help='Encoder hidden dimension')
    p.add('--dec_hid_dim', type=int, default=128, help='Decoder hidden dimension')
    p.add('--block_num', type=int, default=5, help='ResNet block number')

    p.add('--dropout', type=float, default=0.0, help='Dropout rate')




    return p.parse_args()

