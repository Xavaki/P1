import numpy as np
import sys
import os
from PIL import Image
import huffman
from bitarray import bitarray
import pprint
from run_length import encode as rl_encode


class Encoder():

    testblock = np.array([
        [52, 55, 61, 66, 70, 61, 64, 73],
        [63, 59, 55, 90, 109, 85, 69, 72],
        [62, 59, 68, 113, 144, 104, 66, 73],
        [63, 58, 71, 122, 154, 106, 70, 69],
        [67, 61, 68, 104, 126, 88, 68, 70],
        [79, 65, 60, 70, 77, 68, 58, 75],
        [85, 71, 64, 59, 55, 61, 65, 83],
        [87, 79, 69, 68, 65, 76, 78, 94],
    ])

    testimage = np.reshape(np.arange(153).astype(int), (9, 17))

    quantization_tables = {
        "Y": {"50":
              np.array([
                  [16, 11, 10, 16, 24, 40, 51, 61],
                  [12, 12, 14, 19, 26, 58, 60, 55],
                  [14, 13, 16, 24, 40, 57, 69, 56],
                  [14, 17, 22, 29, 51, 87, 80, 62],
                  [18, 22, 37, 56, 68, 109, 103, 77],
                  [24, 36, 55, 64, 81, 104, 113, 92],
                  [49, 64, 78, 87, 103, 121, 120, 101],
                  [72, 92, 95, 98, 112, 100, 103, 99],
              ])},
        "C": {"50":
              np.array([
                  [17, 18, 24, 47, 99, 99, 99, 99],
                  [18, 21, 26, 66, 99, 99, 99, 99],
                  [24, 26, 56, 99, 99, 99, 99, 99],
                  [47, 66, 99, 99, 99, 99, 99, 99],
                  [99, 99, 99, 99, 99, 99, 99, 99],
                  [99, 99, 99, 99, 99, 99, 99, 99],
                  [99, 99, 99, 99, 99, 99, 99, 99],
                  [99, 99, 99, 99, 99, 99, 99, 99],
              ])}
    }

    def dct(self, block, p=False):
        # performs 2D dct-ii on an 8x8x1 block, using the formula in the slides
        # returns: dct coefficients
        def alpha_factor(n):
            return np.sqrt(1/8) if n == 0 else np.sqrt(2/8)

        if p:
            print("Unprocessed block:\n", block)
        # we center the values around zero (sblock ~ shifted_block)
        sblock = block-128
        if p:
            print("8x8 shifted block:\n", sblock)
        out = np.zeros(block.shape, dtype=float)
        for uv, val in np.ndenumerate(sblock):
            u = uv[0]
            v = uv[1]
            aux = np.zeros(block.shape, dtype=float)
            for xy, auxval in np.ndenumerate(aux):
                x = xy[0]
                y = xy[1]
                aux[xy] = sblock[xy] * np.cos((np.pi/8) * (x + 1/2) * u) * \
                    np.cos((np.pi/8) * (y + 1/2) * v)

            out[uv] = alpha_factor(u) * alpha_factor(v) * sum(sum(aux))

        if p:
            print("DCT'd block:\n", out)
        return out

    def quantize(self, block, quantization_table, p=False):
        # quantizes according to quantizer table, which in turn depends on desired compression level
        # returns: 8x8 quantized integer block
        quantized = np.round(block / quantization_table)
        if p:
            print("Quantized block:\n", quantized)
        return quantized

    def encode_block(self, block, p=False):
        # encodes block using run length encoder from the previous exercise
        # returns: encoded bit sequence
        def unravel(block):
            # traverses block in zig-zag pattern and returns as 1D array (blockf ~ blockflippped)
            blockf = np.flip(block, 1)
            n = block.shape[0]
            diag_indices = np.flip(np.arange(-n+1, n))
            diags = []
            for i, d in enumerate(diag_indices):
                if i % 2 != 0:
                    diags += list(blockf.diagonal(d))
                else:
                    diags += list(np.flip(blockf.diagonal(d)))
            return diags

        unraveled = unravel(block)
        if p:
            print("Unraveled block:\n", unraveled)

        rl_encoded = rl_encode(unraveled, p=False)

        if p:
            print("Run length encoded sequence:\n", rl_encoded)

        return rl_encoded

    def image_2_blocks(self, npim, p=False):
        # splits input image into 8x8 blocks
        # zero-pads if dimensions are not multiples of 8 (I realize this is not optimal)
        # returns: sequence of blocks, along with meta information for reconstruction
        def zero_pad(npim):
            n = npim.shape[1]
            aux = np.hstack(
                (npim, np.zeros((npim.shape[0], int(np.ceil(n/8)*8-n)))))
            m = npim.shape[0]
            aux = np.vstack(
                (aux, np.zeros((int(np.ceil(m/8)*8-m), aux.shape[1]))))
            return aux

        def image_split(npim):
            # returns array of 8x8 blocks (left-right->top->bottom)
            blocks = []
            nrows = npim.shape[0]/8
            ncols = npim.shape[1]/8
            rows = np.vsplit(npim, nrows)
            for r in rows:
                for c in np.hsplit(r, ncols):
                    blocks.append(c)

            return blocks

        # npim = np.array(im)
        if p:
            print(f'Input image dims: {npim.shape}\n', npim)

        npim = zero_pad(npim)
        if p:
            print(f'Zero padded image dims: {npim.shape}\n', npim)

        blocks = image_split(npim)
        if p:
            print("Blocked image:")
            for b in blocks:
                print(b)

        meta = {"nblocks": len(blocks), "dims": npim.shape}
        return blocks, meta

    def huffman_encoder(self, seqs, p=False):
        # receives list of run-length encoded blocks and performs huffman encoding on whole sequence
        whole_seq = np.concatenate(seqs)
        symbols, counts = np.unique(whole_seq, return_counts=True)
        symbol_probs = [(s, c/len(whole_seq))
                        for (s, c) in zip(symbols, counts)]
        huff_codebook = huffman.codebook(symbol_probs)

        if p:
            print("Huffman table for image:")
            pprint.pprint(huff_codebook)

        bitarray_huff_codebook = {}
        for (k, v) in huff_codebook.items():
            bitarray_huff_codebook[k] = bitarray(v)

        encoded_seqs = []
        for s in seqs:
            a = bitarray()
            a.encode(bitarray_huff_codebook, s)
            encoded_seqs.append(''.join([str(x) for x in a.tolist()]))

        if p:
            print("Huffman encoded blocks:")
            for s in encoded_seqs:
                print(s)

        return encoded_seqs, huff_codebook

    def chroma_subsample(self, im, n=2, p=False):
        # converts image to ycbcr colorspace and performs 1:n:n subsampling
        # returns: array of (3) image channels
        im = im.convert('YCbCr')
        npim = np.array(im)
        Y_channel = npim[:, :, 0]
        Cb_channel = npim[:, :, 1][0::n, 0::n]
        Cr_channel = npim[:, :, 2][0::n, 0::n]

        if p:
            print("Y channel dims:", Y_channel.shape)
            print("Cb channel dims:", Cb_channel.shape)
            print("Cr channel dims:", Cr_channel.shape)
            # Image.fromarray(Y_channel).show()
            # Image.fromarray(Cb_channel).show()
            # Image.fromarray(Cr_channel).show()

        return Y_channel, Cb_channel, Cr_channel

    def encode(self, imfile, compression_factor="50", p=False):
        # simulates jpeg encoding on input image and saves output to .txt file

        # print(f'encoding {imfile}...')

        def encode_channel(channel, type):
            blocks, meta = self.image_2_blocks(channel)
            qt = self.quantization_tables[type][compression_factor]
            seqs = []
            for b in blocks:
                seqs.append(self.encode_block(
                    self.quantize(self.dct(b), qt)))
            encoded_seqs, huffman_codebook = self.huffman_encoder(seqs)
            return encoded_seqs, huffman_codebook, meta

        def write_channel_info(outfile, seqs, codebook, meta):
            outfile.write(str(meta["nblocks"]) + " * " +
                          str(meta["dims"][0]) + " * " + str(meta["dims"][1]) + "\n\n")
            for ci in codebook.items():
                outfile.write(' * '.join([str(x) for x in ci]) + '\n')
            outfile.write('\n')
            for s in seqs:
                outfile.write(str(s) + "\n")
            outfile.write("\n")

        im = Image.open(imfile)
        Y, Cb, Cr = self.chroma_subsample(im)

        Y_seqs, Y_codebook, Y_meta = encode_channel(Y, 'Y')
        Cb_seqs, Cb_codebook, Cb_meta = encode_channel(Cb, 'C')
        Cr_seqs, Cr_codebook, Cr_meta = encode_channel(Cr, 'C')

        # write to outfile
        out = open(f'{imfile.split(".png")[0]}_jpeg.txt', "w")
        write_channel_info(out, Y_seqs, Y_codebook, Y_meta)
        write_channel_info(out, Cb_seqs, Cb_codebook, Cb_meta)
        write_channel_info(out, Cr_seqs, Cr_codebook, Cr_meta)
        out.close()


testblock = np.array([
    [52, 55, 61, 66, 70, 61, 64, 73],
    [63, 59, 55, 90, 109, 85, 69, 72],
    [62, 59, 68, 113, 144, 104, 66, 73],
    [63, 58, 71, 122, 154, 106, 70, 69],
    [67, 61, 68, 104, 126, 88, 68, 70],
    [79, 65, 60, 70, 77, 68, 58, 75],
    [85, 71, 64, 59, 55, 61, 65, 83],
    [87, 79, 69, 68, 65, 76, 78, 94],
])


def idct(block, p=False):

    # performs 2D dct-iii (inverse) on an 8x8x1 block
    # returns: original block
    def C_factor(n):
        return np.sqrt(1/2) if n == 0 else 1
    # shifted out ~ sout
    sout = np.zeros(block.shape, dtype=float)
    for xy, val in np.ndenumerate(block):
        x = xy[0]
        y = xy[1]
        aux = np.zeros(block.shape, dtype=float)
        for uv, auxval in np.ndenumerate(aux):
            u = uv[0]
            v = uv[1]
            aux[uv] = block[uv] * np.cos((np.pi*(2*x + 1)*u)/16) * \
                np.cos((np.pi*(2*y + 1)*v)/16) * \
                C_factor(u) * C_factor(v)

        sout[xy] = sum(sum(aux)) * 1/4

    # we de-shift the values
    out = sout + 128

    if p:
        print("Original block:\n", out)

    return out


def dequantize(block):
    pass


def decode_seq(seq):

    def ravel(seq):
        n = int(np.sqrt(len(seq)))
        diag_indices = np.flip(np.arange(-n+1, n))

    pass


def blocks_2_image(blocks, meta):
    pass


def huffman_decode(seqs):
    pass


def compose_image(channels, n=2):
    pass


def decode(filename, compression_factor="50"):
    pass


if __name__ == '__main__':

    try:
        mode = sys.argv[1]
        infile = sys.argv[2]
        if mode == "encode" and infile in os.listdir():
            np.set_printoptions(precision=2, suppress=True)
            e = Encoder()
            print(f'encoding {infile}...')
            print("(this might take a while)")
            e.encode(infile)
            print(f'{infile} encoded!')
        else:
            print("Invalid arguments")
    except Exception as e:
        print(f'Invalid arguments {e}')

    exit()
