from __future__ import unicode_literals
import youtube_dl
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import csv
import ast
import os
import os.path
#Class definitions

em_class = { '/m/03j1ly':'Emergency vehicle',
		'/m/04qvtq':'Police car(siren)',
		'/m/012n7d':'Ambulance(siren)',
		'/m/012ndj':'Fire engine, fire truck(siren)',
		'/m/03kmc9':'Siren'		
}
non_em_class = {'/m/07pbtc8':'Walk, footsteps',
		'/m/0gy1t2s':'Bicycle bell',
		'/m/03m9d0z':'Wind',
		'/t/dd00092':'Wind noise(microphone)',
		'/m/0jb2l'  :'Thunderstorm',
		'/m/0ngt1'  :'Thunder',
		'/m/06mb1'  :'Rain',
		'/m/0j2kx'  :'Waterfall',
		##can probably exclude this
		'/m/028v0c' :'Silence',
		'/m/096m7z' :'Noise',
		'/m/06_y0by':'Environmental noise',
		'/m/07rgkc5':'Static',
		'/m/0chx_'  :'White noise',
		'/m/06bz3'  :'Radio',		
		#could be noise classes those listed above
		
		###not sure about this '/m/07yv9'  :'Vehicle',
		###not sure about this '/m/012f08' :'Moto vehicle(road)',
		'/m/0k4j'   :'Car',
		'/m/0912c9' :'Vehicle horn, car horn, honking',
		'/m/07rknqz':'Skidding',
		'/m/0h9mv'  :'Tire squeal',
		'/t/dd00134':'Car passing by',
		'/m/07r04'  :'Truck',
		'/m/05x_td' :'Air horn,truck horn',
		'/m0/4_sv'  :'Motorcycle',
	 	'/m/0btp2'  :'Traffic noise, roadway noise',
		'/m/0195fx' :'Subway, metro, underground',
		'/m/0199g'  :'Bicycle',
		'/m/06_fw'  :'Skateboard',
		
		##very important to classify these as non-em
		'/m/0dgbq'  :'Civil defense siren',
		'/m/030rvx' :'Buzzer',
		'/m/01y3hg' :'Smoke detector, smoke alarm',
		'/m/0c3f7m' :'Fire alarm'				
}

##### Set the relevant paths #####

# Specify paths where to save the downloaded files
# and the paths wher to save the extracted audio files

em_download_path = '/Users/sheel/Downloads/emergency/'
nonem_download_path = '/Users/sheel/Downloads/nonEmergency/'

em_files_path = '/Users/sheel/emergency/'
nonem_files_path = '/Users/sheel/nonEmergency/'

##################################

# mention the name of the .csv file
with open('balanced_train_segments.csv') as csvfile:
	readCSV = csv.reader(csvfile,delimiter=str(u',').encode('utf-8'))
	count=0
	em_c=0
	non_em_c=0
	
	for row in readCSV:
		if(count<3):
			count=count+1;
		else:
			v_id = row[0]
			#v_start = ast.literal_eval(row[1])
			#v_end = ast.literal_eval(row[2])
			v_start = float(row[1])
			v_end  = float(row[2])
			labels_list = row[3:]
			
			labels_id = []
			for labels in labels_list:
				if labels == '':
					break
				else:
					labels = labels.replace('\"','')
					labels = labels.strip()
					labels_id.append(labels)
			
			v_class = []
			for labels in labels_id:
				v_class.extend(labels.split(","))

			flg1=0
			flg2=0
			
			##Excluding videos present in both EM and non-EM
			if(set(v_class)&set(em_class.keys())!=set([])):
				flg1=1
			if(set(v_class)&set(non_em_class.keys())!=set([])):
				flg2=1

			if flg1==1 and flg2==1:
				continue;
			elif flg1==1:
				print v_id,v_start,v_end,'em'
				ydl_opts = {
    					'format': 'bestaudio/best',
					'ignoreerrors':'True',
    					'postprocessors': [{
        					'key': 'FFmpegExtractAudio',
        					'preferredcodec': 'wav',
        					'preferredquality': '192',
    					}],
    					'outtmpl':'emergency/%(id)s.%(ext)s',    
				}

				with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    					ydl.download(['http://www.youtube.com/watch?v='+row[0]])

				input_file = em_download_path+v_id+'.wav'

				# break
				if(os.path.exists(input_file)):
					em_c = em_c +1
					output_file = em_files_path+str(em_c)+'.wav'
					print output_file
					ffmpeg_extract_subclip(input_file,v_start,v_end,targetname=output_file)
					os.remove(input_file)
					
			elif flg2==1:
				print v_id,v_start,v_end,'Non-em'
				ydl_opts = {
    					'format': 'bestaudio/best',
					'ignoreerrors':'True',
    					'postprocessors': [{
        					'key': 'FFmpegExtractAudio',
        					'preferredcodec': 'wav',
        					'preferredquality': '192',
    					}],
    					'outtmpl':'nonEmergency/%(id)s.%(ext)s',    
				}

				with youtube_dl.YoutubeDL(ydl_opts) as ydl:
	    				ydl.download(['http://www.youtube.com/watch?v='+row[0]])
				
				input_file = nonem_download_path+v_id+'.wav'
				 
				if(os.path.exists(input_file)):
					non_em_c = non_em_c+1
					output_file = nonem_files_path+str(non_em_c)+'.wav'
					print output_file
					ffmpeg_extract_subclip(input_file,v_start,v_end,targetname=output_file) 
					os.remove(input_file)
