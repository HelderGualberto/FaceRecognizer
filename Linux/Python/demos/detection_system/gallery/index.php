<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1" />
<title>Gallery from Folder Demo</title>
<style type="text/css">
<!--
li{
    list-style-type:none;
    margin-right:10px;
    margin-bottom:10px;
    float:left;
}

-->
</style></head>

<body>

<ul>
    <?php
        $dirname = "/pad-lsi/openface/SafetyCity/Code/Python/demos/web2/predic-entrada-eletrica/";
        $images = scandir($dirname);
	print_r($images);
	$day = date(d); 
	echo "$day";
	$imagesOfTheDay = preg_grep("/(predic_" . $date . "(.*)/", explode("\n", $images));
	echo "$imagesOfTheDay";
        $ignore = Array(".", "..");
        foreach($imagesOfTheDay as $curimg){
            if(!in_array($curimg, $ignore)) {
                echo "<li><a href='".$dirname.$curimg."'><img src='img.php?src=".$dirname.$curimg."&w=300&zc=1' alt='' /></a></li> ";
            }
        }                 
    ?>
</ul>

</body>
</html>

