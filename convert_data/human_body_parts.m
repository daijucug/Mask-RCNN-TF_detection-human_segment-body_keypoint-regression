#human_body_pars
load('index_ade20k.mat');


strings={'back','head','left arm','left foot','left hand','left leg','left shoulder','neck','right arm','right foot','right hand','right leg','right shoulder','torso'};
N=22210;
  
for n = 1:N
  filename = fullfile(index.folder{n}, index.filename{n});
  [Om, Oi, Pm, Pi, objects, parts] = loadAde20K(filename);

  object_class = objects.class;
  r = rows(objects.class);
  ok=0;
  for i =1:r
    if findstr(object_class{i,1},'person')
      ok=1;
      break
    endif
  end
  
  pndx = setdiff(unique(Pm),0);
  index_object_names = index.objectnames(pndx);
  if ok==0 || isempty(index_object_names)
    continue
  endif
  ok=0;
  for i=1:14
    if any(ismember(index_object_names,strings{i})) != 0
      ok=1;
      break
    endif
  end
  
  if ok ==1
    
    #disp('ok');
    #figure; imshow(Om, []); title('Object classes');
    #colormap(cat(1, [0 0 0], hsv(255)));
    
    #figure; imshow(Oi, []); title('Object classes');
    #colormap(cat(1, [0 0 0], hsv(255)));
    
    #subplot(round(sqrt(Nlevels)), ceil(sqrt(Nlevels)), 1)
    #imshow(Pm(:,:,1), []); title('Part classes')
    #colormap(cat(1, [0 0 0], hsv(255)))
    
    file_Om = sprintf('output_dir/Om%d.mat',n);
    file_Oi = sprintf('output_dir/Oi%d.mat',n);
    file_Pm = sprintf('output_dir/Pm%d.mat',n);
    file_Pi = sprintf('output_dir/Pi%d.mat',n);
    file_objects = sprintf('output_dir/objects%d.mat',n);
    file_parts = sprintf('output_dir/parts%d.mat',n);
    file_name = sprintf('output_dir/file%d.jpg',n);
    
    save(file_Om, 'Om',"-mat7-binary");
    save(file_Oi, 'Oi',"-mat7-binary");
    save(file_Pm, 'Pm',"-mat7-binary");
    save(file_Pi, 'Pi',"-mat7-binary");
    save(file_objects, 'objects',"-mat7-binary");
    save(file_parts, 'parts',"-mat7-binary");
    copyfile(filename,file_name);
    
    pndx = setdiff(unique(Pm),0);
    disp('Parts present in this image:');
    disp(n);
  endif
  #disp('next');
  #fflush(stdout)
  
  
  
  #{
  wndx = setdiff(unique(Om),0);
  disp('Objects present in this image (and their wordnet hierarchy):')
  for i = 1:length(wndx)
      %disp(sprintf('%60s', index.objectnames{wndx(n)}))
      if findstr(index.objectnames{wndx(i)},'person')
        disp('ok')
        figure; imshow(Om, []); title('Object classes')
        colormap(cat(1, [0 0 0], hsv(255)))
      endif
  end
  #}
end
