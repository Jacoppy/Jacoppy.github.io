---
title: "Eumeration for Windows"
---

Here at following a little batch script I created which could be usefull for subdomain discovering and enumeration.
I really love [CommandoVM](https://github.com/fireeye/commando-vm) so yea, why not developing a little [lazyrecon](https://github.com/nahamsec/lazyrecon) for it ;)
Enjoy!

```batch
@echo off
echo       ############################
echo .             CommandoVM Recon         . 
echo       ############################
echo ----------------------------------------
set year=%date:~-4%
set month=%date:~3,2%
set day=%date:~7,2%
set @domain=%1
set @rootpath=%2
set assetfinder= C:\Users\Entropico\go\bin\assetfinder.exe
set aquatonepath= C:\Tools\Aquatone\aquatone.exe
set dnscan=  C:\Tools\dnscan\dnscan.py
set dnscan_wordlist= C:\Tools\SecLists\Discovery\DNS\jhaddix-dns.txt
set sublister= C:\Tools\Sublist3r\sublist3r.py
set amasspath= C:\Tools\Amass\amass.exe
set dirsearchpath=C:\Tools\dirsearch\
set chromiumpath= C:\Tools\Aquatone\chromium\chrome.exe
set waybackurlspath= C:\Users\Entropico\go\bin\waybackurls.exe
set httprobe= C:\Users\Entropico\go\bin\httprobe.exe
set python3 = C:\Python38\python.exe
echo Starting the enumeration of %@domain%
echo Creating Folder for %@domain%
set saveresult=%@rootpath%\%@domain%_%year%_%month%
mkdir %saveresult%
echo Amass is running...
%amasspath% enum -d %@domain% -o %saveresult%\amass.txt --passive
echo Amass Done
echo Sublist3r is running...
python %sublister%  -d "%@domain%" -b -o %saveresult%\sublist3r.txt -v
echo Sublist3r Done
echo DNScan is running...
python %dnscan%  -d %@domain% -w %dnscan_wordlist% -o %saveresult%\dnscan.txt -R 8.8.8.8 -r
awk  '/%@domain%/{print $3}' %saveresult%\dnscan.txt | tail -n+2 > %saveresult%\dnscan_clean.txt
rm %saveresult%\dnscan.txt
echo DNScan Done
echo AssetFinder is running...
%assetfinder%  %@domain%  > %saveresult%\assetfinder.txt
echo AssetFinder Done
echo Merging all subdomains found...
cat  %saveresult%\* | sort | uniq > %saveresult%\final_subdomain_list.txt
echo HTTProbe is running...
cat %saveresult%\final_subdomain_list.txt | %httprobe%  > %saveresult%\final_online_subdomains_temp.txt
cat   %saveresult%\final_online_subdomains_temp.txt  | sed "s/\http\:\/\///g" |  sed "s/\https\:\/\///g" | sort | uniq > %saveresult%\final_online_subdomains.txt
rm %saveresult%\final_online_subdomains_temp.txt
echo HTTProbe Done
echo WayBackURLS is running...
cat %saveresult%\final_online_subdomains.txt | %waybackurlspath% -dates -no-subs > %saveresult%\waybackurls.txt
echo WayBackURLS Done
echo Aquatone Enumerate %@domain%
echo Starting Aquatone...
cat %saveresult%\final_online_subdomains.txt | %aquatonepath% -ports medium -scan-timeout 10000 -out %saveresult%\aquatone-report -screenshot-timeout 6000 -chrome-path %chromiumpath%
echo Aquatone Done
echo Performing Dirsearch...
%python3% %dirsearchpath%dirsearch.py -L %saveresult%\aquatone-report\aquatone_urls.txt -e *
echo Done ! Save every result in %saveresult%
```
