from pytorch3d.io import load_ply
import torch
import h5py, pickle


our_text_file = {}
for i in range(len(self.text_files)):
    key = self.keys[i]
    text_file = self.text_files[key]
    descriptions = text_file.read_text().split('\n')
    descriptions = list(filter(lambda t: len(t) > 0, descriptions))
    if len(descriptions) <= 0:
        print('error')
    our_text_file[i] = descriptions
out_file = open('./abo_text.pkl', 'wb')
pickle.dump(our_text_file, out_file)
out_file.close()


pcs = []
for i in range(len(self.image_files)):
    key = self.keys[i]
    image_file = self.image_files[key]
    pc = load_ply(image_file)[0].unsqueeze(0)
    pcs.append(pc)
pcs_our = torch.cat(pcs)
out_file = h5py.File('abo_pc.h5','w')
out_file['data'] = pcs_our.numpy()
out_file.close()

