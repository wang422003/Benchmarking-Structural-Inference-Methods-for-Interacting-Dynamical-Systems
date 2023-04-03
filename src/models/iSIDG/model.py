from modules import MLPEncoder, MPMEncoder, GINEncoder, GATEncoder, GCNEncoder
from modules import MLPDecoder, MULTIDecoder, RNNDecoder


def model_selection(args):
    if args.encoder == 'mpm':
        encoder = MPMEncoder(
            n_in=args.timesteps * args.dims,
            n_hid=args.encoder_hidden,
            n_out=args.edge_types,
            do_prob=args.encoder_dropout,
            factor=args.factor
        )
    elif args.encoder == 'gin':
        encoder = GINEncoder(
            n_in=args.timesteps * args.dims,
            n_hid=args.encoder_hidden,
            n_out=args.edge_types,
            do_prob=args.encoder_dropout,
            factor=args.factor
        )
    elif args.encoder == 'gcn':
        encoder = GCNEncoder(
            n_in=args.timesteps * args.dims,
            n_hid=args.encoder_hidden,
            n_out=args.edge_types,
            do_prob=args.encoder_dropout,
            factor=args.factor
        )
    elif args.encoder == 'gat':
        encoder = GATEncoder(
            n_feat=args.timesteps,
            n_in=args.timesteps * args.dims,
            n_hid=args.encoder_hidden,
            n_out=args.edge_types,
            do_prob=args.encoder_dropout,
            factor=args.factor
        )
    elif args.encoder == 'mlp':
        encoder = MLPEncoder(
            n_in=args.timesteps * args.dims,
            n_hid=args.encoder_hidden,
            n_out=args.edge_types,
            do_prob=args.encoder_dropout,
            factor=args.factor
        )
    else:
        encoder = None

    if args.decoder == 'mlp':
        decoder = MLPDecoder(
            n_in_node=args.dims,
            edge_types=args.edge_types,
            msg_hid=args.decoder_hidden,
            msg_out=args.decoder_hidden,
            n_hid=args.decoder_hidden,
            do_prob=args.decoder_dropout,
            skip_first=args.skip_first
        )
    elif args.decoder == 'multi':
        decoder = MULTIDecoder(
            n_in_node=args.dims,
            edge_types=args.edge_types,
            msg_hid=args.decoder_hidden,
            msg_out=args.decoder_hidden,
            n_hid=args.decoder_hidden,
            do_prob=args.decoder_dropout,
            skip_first=args.skip_first
        )

    elif args.decoder == 'rnn':
        decoder = RNNDecoder(
            n_in_node=args.dims,
            edge_types=args.edge_types,
            n_hid=args.decoder_hidden,
            do_prob=args.decoder_dropout,
            skip_first=args.skip_first
        )
    else:
        decoder = None
    return encoder, decoder


if __name__ == "__main__":
    print("This is module.py")
