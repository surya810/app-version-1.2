$(document).ready(function() {
    $('form').submit(function(e) {
        e.preventDefault();
        var formData = new FormData(this);
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: formData,
            contentType: false,
            cache: false,
            processData: false,
            success: function(response) {
                $('.container').html(response);
            },
            error: function(error) {
                console.log(error);
            }
        });
    });
});
