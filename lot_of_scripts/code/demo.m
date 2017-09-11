%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demo ADE20K
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Load data
load('ADE20K_2016_may/index_ade20k_2015.mat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Read one image
n=100;
filename = fullfile(index.folder{n}, index.filename{n});
[Om, Oi, Pm, Pi, objects, parts] = loadAde20K(filename);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% To get the segmentation according to objects:
figure; imshow(Om, []); title('Object classes')
colormap(cat(1, [0 0 0], hsv(255)))

% Each pixel is one class. ObjectClassMasks=0 means that the pixel is unlabeled. 
% To get the label in one pixel:
index.objectnames(Om(20,30))

% If you want to separate each object instance, you should use the masks
% inside ObjectInstanceMasks. 
figure; imshow(Oi, []); title('Object instances')
colormap(cat(1, [0 0 0], hsv(255)))

%% Get object names
wndx = setdiff(unique(Om),0);
disp('Objects present in this image (and their wordnet hierarchy):')
for n = 1:length(wndx)
    disp(sprintf('%15s -> %s', index.objectnames{wndx(n)}, index.wordnet_synset{wndx(n)}))
end

% the list is also available inside
objects.class

% and also the object attributes (for many images this list might be empty strings)
objects.listattributes

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Working with object parts. Parts are also stored inside
% 'ObjectClassMasks' and 'ObjectInstanceMasks'.
% To access them:
Nlevels = size(Pm,3);

figure;
for i = 1:Nlevels
    subplot(round(sqrt(Nlevels)), ceil(sqrt(Nlevels)), i)
    imshow(Pm(:,:,i), []); title('Part classes')
    colormap(cat(1, [0 0 0], hsv(255)))
end

% you can get the list of parts present in an image as:
pndx = setdiff(unique(Pm),0);
disp('Parts present in this image:')
disp(index.objectnames(pndx)')

% Example of part from level 1:
part_name = index.objectnames{pndx(1)}; 
object_name = index.objectnames{unique(Om(Pm(:,:,1)==pndx(1)))};
disp(sprintf('"%s" is part of "%s"', part_name, object_name))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Understanding the collection
frequent_ndx = find(index.objectcounts>=100);

figure; 
bar(sort(index.objectcounts(frequent_ndx)));

[~,j] = sort(index.proportionClassIsPart(frequent_ndx));
figure; 
imshow(1+(index.objectPresence(frequent_ndx(j),1:1500)>0)-2*(index.objectIsPart(frequent_ndx(j),1:1500)>0),[])
ylabel('Object class')
xlabel('Image ndx')
