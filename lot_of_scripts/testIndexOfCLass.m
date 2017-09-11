#human_body_pars
load('index_ade20k.mat');


strings={'back','head','left arm','left foot','left hand','left leg','left shoulder','neck','right arm','right foot','right hand','right leg','right shoulder','torso'};
N=22210;
  
n=5
filename = fullfile(index.folder{n}, index.filename{n});
[Om, Oi, Pm, Pi, objects, parts] = loadAde20K(filename);
figure; imshow(Om, []); title('Object classes')
colormap(cat(1, [0 0 0], hsv(255)))
index.objectnames(Om(1100,400))
Om(1100,400)
